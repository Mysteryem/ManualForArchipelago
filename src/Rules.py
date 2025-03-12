import ast
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Literal, Union, Callable, ClassVar, Any, Protocol
from enum import IntEnum

from .Regions import regionMap
from .hooks import Rules
from .Helpers import clamp, is_item_enabled, get_items_with_value, is_option_enabled, get_option_value, convert_string_to_type, format_to_valid_identifier

from BaseClasses import MultiWorld, CollectionState
from Utils import cache_self1
from worlds.AutoWorld import World
from worlds.generic.Rules import set_rule, add_rule, CollectionRule

import re
import math
import inspect
import logging

if TYPE_CHECKING:
    from . import ManualWorld

class LogicErrorSource(IntEnum):
    INFIX_TO_POSTFIX = 1 # includes more closing parentheses than opening (but not the opposite)
    EVALUATE_POSTFIX = 2 # includes missing pipes and missing value on either side of AND/OR
    EVALUATE_STACK_SIZE = 3 # includes missing curly brackets

def construct_logic_error(location_or_region: dict, source: LogicErrorSource) -> KeyError:
    object_type = "location/region"
    object_name = location_or_region.get("name", "Unknown")

    if location_or_region.get("is_region", False) or "starting" in location_or_region or "connects_to" in location_or_region:
        object_type = "region"
    elif "region" in location_or_region or "category" in location_or_region:
        object_type = "location"

    if source == LogicErrorSource.INFIX_TO_POSTFIX:
        source_text = "There may be mismatched parentheses, or other invalid syntax for the requires."
    elif source == LogicErrorSource.EVALUATE_POSTFIX:
        source_text = "There may be missing || around item names, or an AND/OR that is missing a value on one side, or other invalid syntax for the requires."
    elif source == LogicErrorSource.EVALUATE_STACK_SIZE:
        source_text = "There may be missing {} around requirement functions like YamlEnabled() / YamlDisabled(), or other invalid syntax for the requires."
    else:
        source_text = "This requires includes invalid syntax."

    return KeyError(f"Invalid 'requires' for {object_type} '{object_name}': {source_text} (ERROR {source})")


# noinspection PyUnusedLocal
def _always(state: CollectionState):
    """Function used in place of `lambda state: True` to reduce the number of lambdas created."""
    return True


# noinspection PyUnusedLocal
def _never(state: CollectionState):
    """Function used in place of `lambda state: False` to reduce the number of lambdas created."""
    return False


class HookFunction(Protocol):
    """Protocol to represent custom hooks.Rules functions in type hints."""
    def __call__(self,
                 world: World,
                 multiworld: MultiWorld,
                 state: CollectionState,
                 player: int,
                 *args, **kwargs) -> bool | str: ...


class RuleBuilder:
    max_runtime_rule_cache_size: ClassVar[int] = 2048

    world: "ManualWorld"
    multiworld: MultiWorld
    player: int

    # Cache of all static string rules.
    rule_cache: dict[str, CollectionRule]

    # Cache of individual |<item>:<item_count>| rules.
    subrule_cache: dict[str, tuple[CollectionRule, str]]

    # Least Recently Used cache of runtime evaluated string rules, as returned from hook functions.
    runtime_rule_cache: OrderedDict[str, CollectionRule]

    def __init__(self, world: "ManualWorld"):
        self.world = world
        self.multiworld = world.multiworld
        self.player = world.player

        self.rule_cache = {}
        self.subrule_cache = {}
        self.runtime_rule_cache = OrderedDict()

    @staticmethod
    def create_collection_rule_from_ast(original_rule_string: str,
                                        node: ast.BoolOp | ast.Call | ast.Constant, args: dict[str, CollectionRule]
                                        ) -> CollectionRule:
        if isinstance(node, ast.Constant):
            # The only allowed constants are True/False.
            literal_value = node.value
            if literal_value is True:
                return _always
            elif literal_value is False:
                return _never
            else:
                raise RuntimeError(f"Unexpected literal evaluation of {original_rule_string} as"
                                   f"\n{ast.dump(node, indent=4)}"
                                   f"\ninto {literal_value!r}")

        # Create an expression that evaluates to `lambda state: <parsed_ast>`, compatible with CollectionRule typing.
        ast_lambda = ast.Lambda(args=ast.arguments(args=[ast.arg(arg="state")]), body=node)
        expr = ast.Expression(body=ast_lambda)

        # Compile the ast Expression node into a <code> object and then evaluate it to create the lambda.
        rule_func: CollectionRule = eval(compile(ast.fix_missing_locations(expr), "<string>", "eval"), args)

        return rule_func

    @staticmethod
    def convert_req_function_args(func, args: list[str]) -> list[tuple[Any, bool]]:
        parameters = inspect.signature(func).parameters
        knownParameters = ["world", "multiworld", "state", "player"]
        index = -1
        parsed_args = []
        for parameter in parameters.values():
            if parameter.name in knownParameters:
                continue
            index += 1
            target_type = parameter.annotation

            if index < len(args) and args[index] != "":
                value = args[index].strip()
            else:
                if parameter.default is not inspect.Parameter.empty:
                    if index < len(args):
                        parsed_args.append((parameter.default, False))
                    continue
                else:
                    if parameter.annotation is inspect.Parameter.empty:
                        raise ConvertReqFunctionArgsError(f"A call of the \"{func.__name__}\" function in \"{{areaName}}\"'s requirement, asks for a value for its argument \"{parameter.name}\" but it's missing.")
                    else:
                        raise ConvertReqFunctionArgsError(f"A call of the \"{func.__name__}\" function in \"{{areaName}}\"'s requirement, asks for a value of type {target_type} for its argument \"{parameter.name}\" but it's missing.")

            if target_type == str or parameter.annotation is inspect.Parameter.empty: #Don't convert since its already a string or if we don't know the type to convert to
                parsed_args.append((value, False))
                continue

            try:
                arg_tuple = convert_string_to_type(value, target_type)
            except Exception as e:
                raise ConvertReqFunctionArgsError(f"A call of the \"{func.__name__}\" function in \"{{areaName}}\"'s requirement, asks for a value of type {target_type}\nfor its argument \"{parameter.name}\" but its value \"{value}\" cannot be converted to {target_type} \nOriginal Error:'{e}'")

            parsed_args.append(arg_tuple)
        return parsed_args

    def make_function_collection_rule(self, func: HookFunction, func_args: tuple,
                                      args_copy_func: Callable[[tuple], list] | None) -> CollectionRule:
        world = self.world
        player = self.player
        multiworld = self.multiworld

        if func in SIMPLE_FUNCTIONS:
            if args_copy_func is not None:
                raise RuntimeError(f"Error: Simple function {func} had complex arguments")
            else:
                num_args = len(func_args)
                if num_args == 0:
                    def collection_rule(state: CollectionState):
                        return func(world, multiworld, state, player)
                elif num_args == 1:
                    func_arg = func_args[0]

                    def collection_rule(state: CollectionState):
                        return func(world, multiworld, state, player, func_arg)
                else:
                    def collection_rule(state: CollectionState):
                        return func(world, multiworld, state, player, *func_args)
        else:
            if args_copy_func is not None:
                def collection_rule(state: CollectionState):
                    result = func(world, multiworld, state, player, *args_copy_func(func_args))
                    if result is True or result is False:
                        return result
                    else:
                        return self.runtime_rule_string_to_callable(func, str(result))(state)
            else:
                num_args = len(func_args)
                if num_args == 0:
                    def collection_rule(state: CollectionState):
                        result = func(world, multiworld, state, player)
                        if result is True or result is False:
                            return result
                        else:
                            return self.runtime_rule_string_to_callable(func, str(result))(state)
                elif num_args == 1:
                    func_arg = func_args[0]

                    def collection_rule(state: CollectionState):
                        result = func(world, multiworld, state, player, func_arg)
                        if result is True or result is False:
                            return result
                        else:
                            return self.runtime_rule_string_to_callable(func, str(result))(state)
                else:
                    def collection_rule(state: CollectionState):
                        result = func(world, multiworld, state, player, *func_args)
                        if result is True or result is False:
                            return result
                        else:
                            s = str(result)
                            return self.runtime_rule_string_to_callable(func, s)(state)
        return collection_rule

    @cache_self1
    def get_function_from_item(self, item: tuple[str, str]) -> tuple[str, CollectionRule]:
        func_name = item[0]
        func_args = item[1].split(",")
        if func_args == ['']:
            func_args.pop()

        func = globals().get(func_name)

        if func is None:
            func = getattr(Rules, func_name, None)

        if not callable(func):
            raise InvalidFunctionError(func_name)

        parsed_args = RuleBuilder.convert_req_function_args(func, func_args)

        base_args: list[Any] = []
        function_args: list[tuple[int, Callable[[], Any]]] = []
        for i, (arg, is_function) in enumerate(parsed_args):
            if is_function:
                # Using `...` to avoid confusion with `None` which is a valid literal argument. Technically `...` is
                # also a valid literal, but it is unlikely to see normal use.
                base_args.append(...)
                # The argument is constructed by calling a function.
                function_args.append((i, arg))
            else:
                # If no function is provided, then the instance is used as-is. This usually means the instance is
                # immutable, but it could be mutable if one of the function's default arguments is used and that default
                # argument is mutable.
                base_args.append(arg)

        if function_args:
            def args_copy_func():
                args = base_args.copy()
                for i, function in function_args:
                    args[i] = function()
                return args

            new_func_args = ()
        else:
            args_copy_func = None
            new_func_args = tuple(base_args)

        rule = self.make_function_collection_rule(func, new_func_args, args_copy_func)
        return func_name, rule

    def requires_string_to_ast(self, area: dict | tuple[HookFunction, tuple], requires_list: str
                               ) -> tuple[ast.BoolOp | ast.Call | ast.Constant, dict[str, CollectionRule]]:
        player = self.player

        # Once the rule is created, it will be cached under the original requires list string.
        original_requires_list = requires_list

        # todo?: Check that the count of "{" and "}" in requires_list is the same?

        # Replace each hook function call with "}" as a placeholder for the CollectionRule for that hook function.
        function_collection_rules = []
        for item in re.findall(r'\{(\w+)\((.*?)\)\}', requires_list):
            try:
                func_name, rule = self.get_function_from_item(item)
            except InvalidFunctionError as ife:
                if isinstance(area, dict):
                    area_type = "region" if area.get("is_region", False) else "location"
                    area_name = area.get("name", f"unknown with these parameters: {area}")
                else:
                    area_type = "hook function"
                    area_name = getattr(area[0], "__name__", "unknown") + f"{area[1]}"
                raise ValueError(f'Invalid function "{ife.func_name}" in {area_type} "{area_name}".')
            except ConvertReqFunctionArgsError as crfae:
                if isinstance(area, dict):
                    area_name = area.get("name", f"unknown with these parameters: {area}")
                else:
                    area_name = getattr(area[0], "__name__", "unknown") + f"{area[1]}"
                raise Exception(crfae.exception_format_str.format(areaName=area_name))
            function_collection_rules.append(rule)
            # todo: Is there a different character we can use? Can we instead iterate the rule like the function that
            #  builds the ast nodes from the simplified rule?
            # Replace the found function with a single closing curly brace.
            requires_list = requires_list.replace(f"{{{func_name}({item[1]})}}", "}", 1)

        if "{" in requires_list:
            # All functions should have been evaluated by this point. If there are any opening curly braces remaining,
            # then there is a syntax error in the requires string.
            raise construct_logic_error(area, LogicErrorSource.EVALUATE_STACK_SIZE)

        item_collection_rules: list[CollectionRule] = []

        # Replace each "|item|" with "{" as a placeholder for the CollectionRule for that item.
        subrule_cache = self.subrule_cache
        world = self.world
        for item in re.findall(r'\|[^|]+\|', requires_list):
            if item in subrule_cache:
                rule, item_base = subrule_cache[item]
                item_collection_rules.append(rule)
                requires_list = requires_list.replace(item_base, "{", 1)
                continue

            # todo: Put each of these tuples in a list and pass this list through to
            #  parsed_logic_string_to_ast_lambda_body,
            #  so that it can try to combine simple chained or/and,
            #  e.g. |item1| and |item2| and |item3| -> has_all(("item1", "item2", "item3"), player)
            # New sub-rule.
            require_type, item_base, item_name, item_count = get_parts_from_item(item)

            if require_type == "category":
                rule = category_sub_rule(world, player, area, item_name, item_count)
                subrule_cache[item] = rule, item_base
                if rule is _always:
                    requires_list = requires_list.replace(item_base, "1", 1)
                else:
                    item_collection_rules.append(rule)
                    requires_list = requires_list.replace(item_base, "{", 1)
            elif require_type == 'item':
                rule = item_sub_rule(world, player, area, item_name, item_count)
                subrule_cache[item] = rule, item_base
                if rule is _always:
                    requires_list = requires_list.replace(item_base, "1", 1)
                else:
                    item_collection_rules.append(rule)
                    requires_list = requires_list.replace(item_base, "{", 1)
            else:
                # Should never happen because get_parts_from_item() defaults to 'item'
                raise RuntimeError(f"Unexpected require_type: '{require_type}'")

        if "|" in requires_list:
            # All "|<item/category[:count]>|" items should have been replaced with string 'replacement fields' ("{}").
            # If there are any pipes ("|") remaining, then there is a syntax error in the requires string.
            raise construct_logic_error(area, LogicErrorSource.EVALUATE_POSTFIX)

        # Attempt to auto-fix missing open/close parentheses and warn when they are found.
        open_count = requires_list.count("(")
        close_count = requires_list.count(")")
        open_close_difference = open_count - close_count
        if open_close_difference > 0:
            requires_list = requires_list + (")" * open_close_difference)
            import warnings
            warnings.warn(f"Invalid rule '{original_requires_list}' for {area} for player {world.player_name} with game"
                          f" {world.game}. Warning: {open_close_difference} missing close parentheses have been added"
                          f" automatically to the end.")
        elif open_close_difference < 0:
            requires_list = ("(" * (-open_close_difference)) + requires_list
            import warnings
            warnings.warn(f"Invalid rule '{original_requires_list}' for {area} for player {world.player_name} with game"
                          f" {world.game}. Warning: {-open_close_difference} missing open parentheses have been added"
                          f" automatically to the start.")

        if "!" in requires_list:
            # Manual supported logical negation at one point. This was dangerous because it enables users to easily
            # create invalid logic by mistake, where gaining an item would reduce accessibility. Archipelago strictly
            # requires that gaining an item only ever increases accessibility or keeps accessibility the same.
            raise Exception(f"Invalid rule '{original_requires_list}' for {area} for player {world.player_name} with"
                            f" game {world.game}. Error: Rule contains negation ('!'). If you need to check for a yaml"
                            f" option being disabled, use YamlDisabled instead.")

        # If everything is boolean logic and/or constants, and either there are no "0"s or no "1"s, then it is easy to
        # deduce the result.
        if not item_collection_rules and not function_collection_rules:
            if "1" not in requires_list:
                # Must evaluate to False because there is only zeroes
                return ast.Constant(False), {}
            if "0" not in requires_list:
                # Must evaluate to True because there is only ones
                return ast.Constant(True), {}

        # "and"/"AND" with word boundaries and optional singular whitespace on either side -> "&"
        requires_list = re.sub(r'\s?\bAND\b\s?', '&', requires_list, 0, re.IGNORECASE)
        # "or"/"OR" with word boundaries and optional singular whitespace on either side -> "|"
        requires_list = re.sub(r'\s?\bOR\b\s?', '|', requires_list, 0, re.IGNORECASE)
        # Remove any remaining whitespace.
        requires_list = re.sub(r"\s+", "", requires_list)

        # Ensure the characters in the string have been reduced to only what is allowed/expected ("&|(){}10").
        # & and | are boolean logic.
        # ( and ) are any parentheses in place from the original rule.
        # { signifies a CollectionRule in item_collection_rules.
        # } signifies a CollectionRule in function_collection_rules.
        # 1 and 0 can come from pre-resolved functions, and 0 can come from a potentially invalid rule.
        used_characters = set(requires_list)
        if used_characters | ALLOWED_PARSED_RULE_CHARACTERS != ALLOWED_PARSED_RULE_CHARACTERS:
            bad_characters = used_characters - ALLOWED_PARSED_RULE_CHARACTERS
            raise Exception(f"Invalid rule '{original_requires_list}' for {area} for player {world.player_name} with"
                            f" game {world.game}. Error: Found unexpected characters after parsing: {bad_characters} in"
                            f" '{requires_list}'.")

        # The item CollectionRules are identified by the order they are found in the rule, from left to right.
        item_collection_rule_names = [f"f{i}" for i in range(len(item_collection_rules))]
        # Reverse the list to create a new list that can be popped like a stack, where the first element popped is the
        # first element in `item_collection_rule_names`.
        item_collection_rule_names_stack = item_collection_rule_names[::-1]

        # Start the function CollectionRule names from the next number after the last item CollectionRule name.
        start = len(item_collection_rules)
        end = len(function_collection_rules) + len(item_collection_rules)
        function_collection_rule_names = [f"f{i}" for i in range(start, end)]
        function_collection_rule_names_stack = function_collection_rule_names[::-1]

        # Further parse the parsed logic string into ast nodes.
        parsed_ast = parsed_logic_string_to_ast_lambda_body(requires_list,
                                                            item_collection_rule_names_stack,
                                                            function_collection_rule_names_stack)

        # The resulting ast node should usually be a BoolOp, but can also be a Call if the rule contains a single
        # function or a Constant (True/False) if the rule had some pre-resolved parts and could be reduced to a
        # constant, e.g. `<complex rule> or True` can be reduced to `True`.
        assert isinstance(parsed_ast, (ast.BoolOp, ast.Call, ast.Constant))

        # The item_collection_rules are referenced by name within `parsed_ast`. Some may have been removed from
        # `parsed_ast` by optimizations, so will be unused in that case.
        args = dict(zip(
            item_collection_rule_names + function_collection_rule_names,
            item_collection_rules + function_collection_rules,
            strict=True))

        return parsed_ast, args

    def static_rule_string_to_callable(self, area: dict, requires_list: str) -> CollectionRule:
        """Fully cached evaluation of static string rules as defined in locations.json."""
        rule_cache = self.rule_cache
        if requires_list in rule_cache:
            return rule_cache[requires_list]

        parsed_ast, args = self.requires_string_to_ast(area, requires_list)

        rule = RuleBuilder.create_collection_rule_from_ast(requires_list, parsed_ast, args)

        # Store the CollectionRule into the cache.
        rule_cache[requires_list] = rule
        return rule

    def runtime_rule_string_to_callable(self, area: dict | HookFunction, requires_list: str) -> CollectionRule:
        """Runtime evaluation of string rules returned by hook functions."""
        if requires_list in self.rule_cache:
            # The runtime rule already exists as part of the static rules, so return the static rule.
            return self.rule_cache[requires_list]

        if requires_list in self.runtime_rule_cache:
            # The runtime rule has already been created and cached. Get the rule and move it to the end.
            self.runtime_rule_cache.move_to_end(requires_list)
            return self.runtime_rule_cache[requires_list]

        parsed_ast, args = self.requires_string_to_ast(area, requires_list)

        rule = RuleBuilder.create_collection_rule_from_ast(requires_list, parsed_ast, args)

        if len(self.runtime_rule_cache) >= RuleBuilder.max_runtime_rule_cache_size:
            # Pop the item at the front (least recently used)
            self.runtime_rule_cache.popitem(False)

        # Add the new rule. The key is not already present, so the item will go to the end of the dict.
        self.runtime_rule_cache[requires_list] = rule

        return rule


class InvalidFunctionError(RuntimeError):
    """Raised when a parsed function is not valid. Often because there is no function found with the specified name."""

    func_name: str

    def __init__(self, func_name: str, *args):
        super().__init__(*args)
        self.func_name = func_name


class ConvertReqFunctionArgsError(RuntimeError):
    """Raised when the parsed arguments for a function do not match the function's signature."""
    exception_format_str: str
    """String to be formatted with `areaName` as a parameter"""

    def __init__(self, exception_format_str: str, *args):
        super().__init__(*args)
        self.exception_format_str = exception_format_str


ALLOWED_PARSED_RULE_CHARACTERS = frozenset("&|(){}10")


def parsed_logic_string_to_ast_lambda_body(parsed_logic_string: str, collection_rule_name_stack: list[str],
                                           function_collection_rule_names_stack: list[str]):
    """
    Manual logic treats "and" and "or" as having the same precedence, so both are evaluated from left-to-right.

    This function iterates a parsed logic string left-to-right, into ast nodes suitable as a lambda body.
    """
    enumerated_parsed_logic_string_iter = enumerate(parsed_logic_string)
    left_operand = None
    operator: Literal["and", "or", None] = None
    for char_index, char in enumerated_parsed_logic_string_iter:
        if char == "(":
            # Iterate further through the string to find the corresponding close parenthesis.
            open_count = 1
            for char_index2, char2 in enumerated_parsed_logic_string_iter:
                if char2 == "(":
                    # Found another open parenthesis, so an extra close parenthesis will need to be found.
                    open_count += 1
                elif char2 == ")":
                    open_count -= 1
                    if open_count == 0:
                        # The close parenthesis has been found, so the start parenthesis is at index `char_index` and
                        # the close parenthesis is at index `char_index2`.
                        break
            else:
                # Normal use checks for a mismatched number of open and close parenthesis, so this conditional branch
                # should be unreachable.
                raise RuntimeError("Error: Missing closing parenthesis")
            # Get everything inside the parentheses.
            parenthesized_operand = parsed_logic_string[char_index + 1:char_index2]
            # Recursively convert the parenthesized contents to ast nodes.
            right_operand = parsed_logic_string_to_ast_lambda_body(parenthesized_operand,
                                                                   collection_rule_name_stack,
                                                                   function_collection_rule_names_stack)
        elif char == ")":
            raise RuntimeError("Error: Missing opening parenthesis")
        elif char == "{":
            # Get the name of the collection rule and construct a Call node to call the collection rule with a "state"
            # argument.
            func_name = collection_rule_name_stack.pop()
            func_args = [ast.Name(id="state", ctx=ast.Load())]
            right_operand = ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), args=func_args)
        elif char == "}":
            # Get the name of the collection rule and construct a Call node to call the collection rule with a "state"
            # argument.
            func_name = function_collection_rule_names_stack.pop()
            func_args = [ast.Name(id="state", ctx=ast.Load())]
            right_operand = ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), args=func_args)
        elif char == "&":
            if operator is not None:
                # todo: Use `construct_logic_error()`
                raise RuntimeError("Error: Multiple operators found in a row")
            operator = "and"
            # Need to find the next operand to apply the operator.
            continue
        elif char == "|":
            if operator is not None:
                # todo: Use `construct_logic_error()`
                raise RuntimeError("Error: Multiple operators found in a row")
            operator = "or"
            # Need to find the next operand to apply the operator.
            continue
        elif char == "1":
            right_operand = ast.Constant(True)
        elif char == "0":
            right_operand = ast.Constant(False)
        else:
            # Normal use checks that all characters in `s` are also in `ALLOWED_PARSED_RULE_CHARACTERS`, so this
            # conditional branch should be unreachable in normal use.
            raise RuntimeError(f"Error: Unexpected character {char!r} at {parsed_logic_string!r}[{char_index}]")

        # Assigning to `operator` should continue to the next character. Everything else should either assign to
        # `right_operand` or raise an exception, so `right_operand` should always be assigned to something non-None.
        assert right_operand is not None

        if left_operand is None:
            left_operand = right_operand
            continue

        # Combine `left_operand` and `right_operand` according to `operator`.
        if operator is None:
            raise RuntimeError("Error: Missing operator")

        if operator == "and":
            if isinstance(left_operand, ast.Constant):
                constant_value = left_operand.value
                if constant_value is False:
                    # `False and right_operand` is always False, so drop `right_operand`.
                    pass
                elif constant_value is True:
                    # `True and right_operand` is always `right_operand`, so drop `left_operand`.
                    left_operand = right_operand
                else:
                    raise RuntimeError(f"Unexpected constant: {constant_value}")
            elif isinstance(right_operand, ast.Constant):
                constant_value = right_operand.value
                if constant_value is False:
                    # `left_operand and False` is always False, so drop `left_operand`.
                    left_operand = right_operand
                elif constant_value is True:
                    # `left_operand and True` is always `left_operand`, so drop `right_operand`.
                    pass
                else:
                    raise RuntimeError(f"Unexpected constant: {constant_value}")
            else:
                left_operand = ast.BoolOp(op=ast.And(), values=[left_operand, right_operand])
        else:
            assert operator == "or"
            if isinstance(left_operand, ast.Constant):
                constant_value = left_operand.value
                if constant_value is True:
                    # `True or right_operand` is always True, so drop `right_operand`.
                    pass
                elif constant_value is False:
                    # `False or right_operand` is always `right_operand`, so drop `left_operand`
                    left_operand = right_operand
                else:
                    raise RuntimeError(f"Unexpected constant: {constant_value}")
            elif isinstance(right_operand, ast.Constant):
                constant_value = right_operand.value
                if constant_value is True:
                    # `left_operand or True` is always True, so drop `left_operand`.
                    left_operand = right_operand
                elif constant_value is False:
                    # `left_operand or False` is always `left_operand`, so drop `right_operand`.
                    pass
                else:
                    raise RuntimeError(f"Unexpected constant: {constant_value}")
            else:
                left_operand = ast.BoolOp(op=ast.Or(), values=[left_operand, right_operand])
        # Reset the operator.
        operator = None
        # Del `right_operand` for safety because accidentally bleeding the previous `right_operand` into the next
        # iteration could result in confusing errors.
        del right_operand

    # Everything combines into the left-most node.
    if left_operand is None:
        # Treat an empty rule as True. Usually, rule parsing should be able to short-circuit an empty rule before
        # calling this function.
        return ast.Constant(True)
    else:
        return left_operand


def get_parts_from_item(item: str):
    """Parses an item in a list returned by item_items_from_requires_list."""
    require_type = 'item'

    if '|@' in item:
        require_type = 'category'

    item_base = item
    item = item.lstrip('|@$').rstrip('|')

    item_parts = item.split(":")  # type: list[str]
    item_name = item
    item_count = "1"

    if len(item_parts) > 1:
        item_name = item_parts[0].strip()
        item_count = item_parts[1].strip()

    return require_type, item_base, item_name, item_count


def category_sub_rule(world: "ManualWorld", player: int, area: dict, item_name: str, item_count: str) -> CollectionRule:
    """Convert "|@<item_name>:<item_count>|" logic into a CollectionRule."""
    category_item_names: list[str] = [item["name"] for item in world.item_name_to_item.values()
                                      if "category" in item and item_name in item["category"]]
    item_count_lower = item_count.lower()
    if item_count_lower == 'all':
        def has_category_count(state: CollectionState):
            items_counts = world.get_item_counts(player)
            category_items_counts = sum([items_counts.get(item_name, 0) for item_name in category_item_names])
            return state.has_from_list(category_item_names, player, category_items_counts)
    elif item_count_lower == 'half':
        def has_category_count(state: CollectionState):
            items_counts = world.get_item_counts(player)
            category_items_counts = sum([items_counts.get(item_name, 0) for item_name in category_item_names])
            return state.has_from_list(category_item_names, player, category_items_counts // 2)
    elif item_count.endswith("%") and len(item_count) > 1:
        percent = float(item_count[:-1]) / 100
        percent = min(percent, 1.0)
        percent = max(0.0, percent)

        def has_category_count(state: CollectionState):
            items_counts = world.get_item_counts(player)
            category_items_counts = sum([items_counts.get(item_name, 0) for item_name in category_item_names])
            required_count = math.ceil(category_items_counts * percent)
            return state.has_from_list(category_item_names, player, required_count)
    else:
        try:
            required_count = int(item_count)
        except ValueError as e:
            raise ValueError(f"Invalid item count `{item_name}` in {area}.") from e

        # todo: It would be preferable if the rule could be replaced entirely, maybe to "1" (True).
        if required_count == 0:
            return _always

        def has_category_count(state: CollectionState):
            return state.has_from_list(category_item_names, player, required_count)

    return has_category_count


def item_sub_rule(world: "ManualWorld", player: int, area: dict, item_name: str, item_count: str) -> CollectionRule:
    """Convert "|<item_name>:<item_count>|" logic into a CollectionRule."""
    item_count_lower = item_count.lower()
    if item_count_lower == 'all':
        def has_item_count(state: CollectionState):
            items_counts = world.get_item_counts(player)
            item_current_count = items_counts.get(item_name, 0)
            return state.has(item_name, player, item_current_count)
    elif item_count_lower == 'half':
        def has_item_count(state: CollectionState):
            items_counts = world.get_item_counts(player)
            item_current_count = items_counts.get(item_name, 0)
            return state.has(item_name, player, item_current_count // 2)
    elif item_count.endswith("%") and len(item_count) > 1:
        percent = float(item_count[:-1]) / 100
        percent = min(percent, 1.0)
        percent = max(0.0, percent)

        def has_item_count(state: CollectionState):
            items_counts = world.get_item_counts(player)
            item_current_count = items_counts.get(item_name, 0)
            item_percent_count = math.ceil(item_current_count * percent)
            return state.has(item_name, player, item_percent_count)
    else:
        try:
            required_item_count = int(item_count)
        except ValueError as e:
            raise ValueError(f"Invalid item count `{item_name}` in {area}.") from e

        # todo: It would be preferable if the rule could be replaced entirely, maybe to "1" (True).
        if required_item_count == 0:
            return _always

        def has_item_count(state: CollectionState):
            return state.has(item_name, player, required_item_count)
    return has_item_count


def set_rules(world: "ManualWorld", multiworld: MultiWorld, player: int):
    rule_builder = RuleBuilder(world)

    # this is only called when the area (think, location or region) has a "requires" field that is a string
    def checkRequireStringForArea(state: CollectionState, area: dict):
        requires_list = area["requires"]

        if requires_list == "":
            return True

        func = rule_builder.static_rule_string_to_callable(area, requires_list)
        return func(state)

    # this is only called when the area (think, location or region) has a "requires" field that is a dict
    def checkRequireDictForArea(state: CollectionState, area: dict):
        canAccess = True

        for item in area["requires"]:
            # if the require entry is an object with "or" or a list of items, treat it as a standalone require of its own
            if (isinstance(item, dict) and "or" in item and isinstance(item["or"], list)) or (isinstance(item, list)):
                canAccessOr = True
                or_items = item

                if isinstance(item, dict):
                    or_items = item["or"]

                for or_item in or_items:
                    or_item_parts = or_item.split(":")
                    or_item_name = or_item
                    or_item_count = 1

                    if len(or_item_parts) > 1:
                        or_item_name = or_item_parts[0]
                        or_item_count = int(or_item_parts[1])

                    if not state.has(or_item_name, player, or_item_count):
                        canAccessOr = False

                if canAccessOr:
                    canAccess = True
                    break
            else:
                item_parts = item.split(":")
                item_name = item
                item_count = 1

                if len(item_parts) > 1:
                    item_name = item_parts[0]
                    item_count = int(item_parts[1])

                if not state.has(item_name, player, item_count):
                    canAccess = False

        return canAccess

    # handle any type of checking needed, then ferry the check off to a dedicated method for that check
    def fullLocationOrRegionCheck(state: CollectionState, area: dict):
        # if it's not a usable object of some sort, default to true
        if not area:
            return True

        # don't require the "requires" key for locations and regions if they don't need to use it
        if "requires" not in area.keys():
            return True

        if isinstance(area["requires"], str):
            return checkRequireStringForArea(state, area)
        else:  # item access is in dict form
            return checkRequireDictForArea(state, area)

    used_location_names = []
    # Region access rules
    for region in regionMap.keys():
        used_location_names.extend([l.name for l in multiworld.get_region(region, player).locations])
        if region != "Menu":
            for exitRegion in multiworld.get_region(region, player).exits:
                def fullRegionCheck(state: CollectionState, region=regionMap[region]):
                    return fullLocationOrRegionCheck(state, region)

                add_rule(world.get_entrance(exitRegion.name), fullRegionCheck)
            entrance_rules = regionMap[region].get("entrance_requires", {})
            for e in entrance_rules:
                entrance = world.get_entrance(f'{e}To{region}')
                add_rule(entrance, lambda state, rule={"requires": entrance_rules[e]}: fullLocationOrRegionCheck(state, rule))
            exit_rules = regionMap[region].get("exit_requires", {})
            for e in exit_rules:
                exit = world.get_entrance(f'{region}To{e}')
                add_rule(exit, lambda state, rule={"requires": exit_rules[e]}: fullLocationOrRegionCheck(state, rule))

    # Location access rules
    for location in world.location_table:
        if location["name"] not in used_location_names:
            continue

        locFromWorld = multiworld.get_location(location["name"], player)

        locationRegion = regionMap[location["region"]] if "region" in location else None

        if locationRegion:
            locationRegion['name'] = location['region']
            locationRegion['is_region'] = True

        if "requires" in location: # Location has requires, check them alongside the region requires
            def checkBothLocationAndRegion(state: CollectionState, location=location, region=locationRegion):
                locationCheck = fullLocationOrRegionCheck(state, location)
                regionCheck = True # default to true unless there's a region with requires

                if region:
                    regionCheck = fullLocationOrRegionCheck(state, region)

                return locationCheck and regionCheck

            set_rule(locFromWorld, checkBothLocationAndRegion)
        elif "region" in location: # Only region access required, check the location's region's requires
            def fullRegionCheck(state, region=locationRegion):
                return fullLocationOrRegionCheck(state, region)

            set_rule(locFromWorld, fullRegionCheck)
        else: # No location region and no location requires? It's accessible.
            def allRegionsAccessible(state):
                return True

            set_rule(locFromWorld, allRegionsAccessible)

    # Victory requirement
    multiworld.completion_condition[player] = lambda state: state.has("__Victory__", player)


def ItemValue(world: World, multiworld: MultiWorld, state: CollectionState, player: int, valueCount: str, skipCache: bool = False) -> bool:
    """When passed a string with this format: 'valueName:int',
    this function will check if the player has collect at least 'int' valueName worth of items\n
    eg. {ItemValue(Coins:12)} will check if the player has collect at least 12 coins worth of items\n
    You can add a second string argument to disable creating/checking the cache like this:
    '{ItemValue(Coins:12,Disable)}' it can be any string you want
    """

    valueCount = valueCount.split(":")
    if not len(valueCount) == 2 or not valueCount[1].isnumeric():
        raise Exception(f"ItemValue needs a number after : so it looks something like 'ItemValue({valueCount[0]}:12)'")
    value_name = valueCount[0].lower().strip()
    requested_count = int(valueCount[1].strip())

    if not hasattr(world, 'itemvalue_rule_cache'): #Cache made for optimization purposes
        world.itemvalue_rule_cache = {}

    if not world.itemvalue_rule_cache.get(player, {}):
        world.itemvalue_rule_cache[player] = {}

    if not skipCache:
        if not world.itemvalue_rule_cache[player].get(value_name, {}):
            world.itemvalue_rule_cache[player][value_name] = {
                'state': {},
                'count': -1,
                }

    if (skipCache or world.itemvalue_rule_cache[player][value_name].get('count', -1) == -1
            or world.itemvalue_rule_cache[player][value_name].get('state') != dict(state.prog_items[player])):
        # Run First Time, if state changed since last check or if skipCache has a value
        existing_item_values = get_items_with_value(world, multiworld, value_name)
        total_Count = 0
        for name, value in existing_item_values.items():
            count = state.count(name, player)
            if count > 0:
                total_Count += count * value
        if skipCache:
            return total_Count >= requested_count
        world.itemvalue_rule_cache[player][value_name]['count'] = total_Count
        world.itemvalue_rule_cache[player][value_name]['state'] = dict(state.prog_items[player])
    return world.itemvalue_rule_cache[player][value_name]['count'] >= requested_count

# Two useful functions to make require work if an item is disabled instead of making it inaccessible
def OptOne(world: World, multiworld: MultiWorld, state: CollectionState, player: int, item: str, items_counts: Optional[dict] = None):
    """Check if the passed item (with or without ||) is enabled, then this returns |item:count|
    where count is clamped to the maximum number of said item in the itempool.\n
    Eg. requires: "{OptOne(|DisabledItem|)} and |other items|" become "|DisabledItem:0| and |other items|" if the item is disabled.
    """
    if item == "":
        return "" #Skip this function if item is left blank
    if not items_counts:
        items_counts = world.get_item_counts()

    require_type = 'item'

    if '@' in item[:2]:
        require_type = 'category'

    item = item.lstrip('|@$').rstrip('|')

    item_parts = item.split(":")
    item_name = item
    item_count = '1'

    if len(item_parts) > 1:
        item_name = item_parts[0]
        item_count = item_parts[1]

    # todo: If item_count is 0, return "1" or "" because the result is always True, or just return `True`?
    if require_type == 'category':
        if item_count.isnumeric():
            #Only loop if we can use the result to clamp
            category_items = [item for item in world.item_name_to_item.values() if "category" in item and item_name in item["category"]]
            category_items_counts = sum([items_counts.get(category_item["name"], 0) for category_item in category_items])
            item_count = clamp(int(item_count), 0, category_items_counts)
        return f"|@{item_name}:{item_count}|"
    elif require_type == 'item':
        if item_count.isnumeric():
            item_current_count = items_counts.get(item_name, 0)
            item_count = clamp(int(item_count), 0, item_current_count)
        return f"|{item_name}:{item_count}|"

# OptAll check the passed require string and loop every item to check if they're enabled,
def OptAll(world: World, multiworld: MultiWorld, state: CollectionState, player: int, requires: str):
    """Check the passed require string and loop every item to check if they're enabled,
    then returns the require string with items counts adjusted using OptOne\n
    eg. requires: "{OptAll(|DisabledItem| and |@CategoryWithModifedCount:10|)} and |other items|"
    become "|DisabledItem:0| and |@CategoryWithModifedCount:2| and |other items|" """
    requires_list = requires

    items_counts = world.get_item_counts()

    functions = {}
    if requires_list == "":
        return True
    for item in re.findall(r'\{(\w+)\(([^)]*)\)\}', requires_list):
        #so this function doesn't try to get item from other functions, in theory.
        func_name = item[0]
        functions[func_name] = item[1]
        requires_list = requires_list.replace("{" + func_name + "(" + item[1] + ")}", "{" + func_name + "(temp)}")
    # parse user written statement into list of each item
    for item in re.findall(r'\|[^|]+\|', requires):
        itemScanned = OptOne(world, multiworld, state, player, item, items_counts)
        requires_list = requires_list.replace(item, itemScanned)

    for function in functions:
        requires_list = requires_list.replace("{" + function + "(temp)}", "{" + func_name + "(" + functions[func_name] + ")}")
    return requires_list

# Rule to expose the can_reach_location core function
def canReachLocation(world: World, multiworld: MultiWorld, state: CollectionState, player: int, location: str):
    """Can the player reach the given location?"""
    if state.can_reach_location(location, player):
        return True
    return False

def YamlEnabled(world: "ManualWorld", multiworld: MultiWorld, state: CollectionState, player: int, param: str) -> bool:
    """Is a yaml option enabled?"""
    return is_option_enabled(multiworld, player, param)

def YamlDisabled(world: "ManualWorld", multiworld: MultiWorld, state: CollectionState, player: int, param: str) -> bool:
    """Is a yaml option disabled?"""
    return not is_option_enabled(multiworld, player, param)


SIMPLE_FUNCTIONS: frozenset[Callable] = frozenset({ItemValue, canReachLocation, YamlEnabled, YamlDisabled})
