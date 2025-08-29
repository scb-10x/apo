import copy
from collections import Counter
import re

class Executor: 
    """
        A wrapper for function calls. 
        It implements the function calls by calling `execute` in function_calls,
        and records the function call names and args for evaluation purposes. 
    
        Notes: 
            1. This executor is a sample one. It will only check whether the output matches the gold functions. 
               If it matches, we will return the gold return values. 
               It it does not match, it will return nothing. 
               However, in real evaluations, the executor will return adequate values even though it is not an exact match with gold functions. 
            2. Please do not try to tamper with attributes in the Executor. Doing so will lead to errors. 
    """
    def __init__(self, tool_registry, action_registry, gold_functions, threshold=0.4):
        self.function_call_stats = []
        self.tool_registry = tool_registry
        self.action_registry = action_registry
        self.gold_functions = gold_functions
        self.threshold = 0.4
        # This is a temporary value. The value may be subject to change by the organizers. 
    
    def execute(self, function_list): 
        """
            Execute the list of functions by checking the gold functions.
            It will also record the function call names and args (for evaluation purposes). 
        """
        copy_functions = copy.deepcopy(function_list)
        for func_item in copy_functions:
            self.function_call_stats.append(copy.deepcopy(func_item))

            # if it matches a gold function
            gold_func_index = self.check_exact_match_gold(func_item)
            if gold_func_index != -1:
                # matches, we return the gold return value
                func_item['return'] = copy.deepcopy(self.gold_functions[gold_func_index]['return'])
            else:
                func_item['return'] = [{'information': 'n/a'}]
            
        return copy_functions
    
    def check_exact_match_gold(self, func_item):
        """
            Checks if the func_item matches any of the gold functions.
            If yes, return the matching index. 
            If not, return -1.  
        """
        for i, gold_function in enumerate(self.gold_functions):
            if gold_function['name'] == func_item['name']:
                # function call name match
                if 'check' in func_item['name']: 
                    # exact match
                    gold_function_param_name_list = list(gold_function['parameters'].keys())
                    gold_function_param_name_list.sort()
                    generated_func_param_name_list = list(func_item['parameters'].keys())
                    generated_func_param_name_list.sort() 
                    if gold_function_param_name_list == generated_func_param_name_list:
                        # param names match
                        gold_function_param_value_list = [gold_function["parameters"][param_name].lower() for param_name in gold_function_param_name_list]
                        generated_func_param_value_list = [func_item["parameters"][param_name].lower() for param_name in generated_func_param_name_list]
                        if gold_function_param_value_list == generated_func_param_value_list:
                            return i 
                elif 'search' in func_item['name']:
                    if self.search_function_match(func_item['parameters'], gold_function):
                        return i

        return -1
    
    def search_function_match(self, pred_function_args, gold_function):
        gold_function_args = gold_function['parameters']
        gold_function_response = gold_function['return']

        # some arguments require exact match because they involve numbers
        repl_args = ['reward', 'price']
        exact_args = ['reward', 'price', 'attack']

        pred_exact_info = ""
        pred_info = ""

        for key in pred_function_args:
            value = ""
            if pred_function_args[key] != "":
                value = pred_function_args[key]
                if not "operator" in key and any([r in key for r in repl_args]):
                    # check if the value is a 'repl_args' 
                    value = value.replace(",", "")
                    value = value.replace(" ", "")
                    value = value.replace("Gold", "")
                    value = value.replace("G", "")
                    value = value.replace("gold", "")
                    value = value.replace("g", "")
            
            if value != "":
                if not "operator" in key and any([e in key for e in exact_args]):
                    if pred_exact_info == "":
                        pred_exact_info = key + "_" + value
                    else:
                        pred_exact_info = pred_exact_info + " " + key + "_" + value
                else:
                    if pred_info == "":
                        pred_info = key + " " + value
                    else:
                        pred_info = pred_info + " " + key + " " + value

        gold_exact_info = ""
        gold_info = ""
        for key in gold_function_args:
            value = gold_function_args[key]
            if not "operator" in key and any([r in key for r in repl_args]):
                value = value.replace(",", "")
                value = value.replace(" ", "")
                value = value.replace("Gold", "")
                value = value.replace("G", "")
                value = value.replace("gold", "")
                value = value.replace("g", "")

            if not "operator" in key and any([e in key for e in exact_args]):
                if gold_exact_info == "":
                    gold_exact_info = key + "_" + value
                else:
                    gold_exact_info = gold_exact_info + " " + key + "_" + value
            else:
                if gold_info == "":
                    gold_info = key + " " + value
                else:
                    gold_info = gold_info + " " + key + " " + value
        
        # exact match of exact arguments
        exact_check = word_f1(pred_exact_info, gold_exact_info)
        if exact_check != 1.0:
            return False

        ret = []
        judgment_score = word_f1(pred_info, gold_info)
        if judgment_score > self.threshold:
            return True
        else:
            return False





def word_f1(pred_item: str, gold_item: str, expose_p_and_r: bool = False) -> float:
    if pred_item is None or gold_item is None:
        return 0
    p_tokens = re.split(r'[ |]+', pred_item.lower())
    g_tokens = re.split(r'[ |]+', gold_item.lower())
    
    common = Counter(g_tokens) & Counter(p_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        if expose_p_and_r:
            return 0, 0, 0
        else:
            return 0
    precision = 1.0 * num_same / len(p_tokens)
    recall = 1.0 * num_same / len(g_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    if expose_p_and_r:
        return precision, recall, f1
    else:
        return f1
