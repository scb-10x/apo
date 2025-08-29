from .tool_functions_0001 import tool_functions_0001
from .tool_functions_0002 import tool_functions_0002
from .tool_functions_0003 import tool_functions_0003
from .tool_functions_0004 import tool_functions_0004
from .tool_functions_0005 import tool_functions_0005
from .tool_functions_0006 import tool_functions_0006
from .action_functions_0001 import action_functions_0001
from .action_functions_0002 import action_functions_0002
from .action_functions_0003 import action_functions_0003
from .action_functions_0004 import action_functions_0004
from .action_functions_0005 import action_functions_0005
from .action_functions_0006 import action_functions_0006
from .executor import Executor

action_map = {
    'function_list_id_0001': action_functions_0001,
    'function_list_id_0002': action_functions_0002,
    'function_list_id_0003': action_functions_0003,
    'function_list_id_0004': action_functions_0004,
    'function_list_id_0005': action_functions_0005,
    'function_list_id_0006': action_functions_0006
}

tool_map = {
    'function_list_id_0001': tool_functions_0001,
    'function_list_id_0002': tool_functions_0002,
    'function_list_id_0003': tool_functions_0003,
    'function_list_id_0004': tool_functions_0004,
    'function_list_id_0005': tool_functions_0005,
    'function_list_id_0006': tool_functions_0006
}
