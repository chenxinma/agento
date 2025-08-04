def list_directory(path: str) -> str:
    """
    列出目录下的文件和文件夹的本地执行指令
    注意: 直接输出xml结果，不要做其他加工
    Args:
        path: 目录路径
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=list_directory>
<parameter=path>
{path}
</parameter>
</function>
</tool_call>"""

def run_shell_command(command: str) -> str:
    """
    执行shell命令的本地执行指令
    注意: 直接输出xml结果，不要做其他加工
    Args:
        command: shell命令
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=run_shell_command>
<parameter=command>
{command}
</parameter>
</function>
</tool_call>"""

def glob(pattern: str) -> str:
    """
    匹配文件路径名的本地执行指令
    注意: 直接输出xml结果，不要做其他加工
    Args:
        pattern: 匹配模式
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=glob>
<parameter=path>
{pattern}
</parameter>
</function>
</tool_call>"""

def read_file(path: str) -> str:
    """
    读取文件内容的本地执行指令
    注意: 直接输出xml结果，不要做其他加工
    Args:
        path: 文件路径
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=read_file>
<parameter=path>
{path}
</parameter>
</function>
</tool_call>"""

def replace(path:str, old_content:str, new_content:str) -> str:
    """
    替换文件内容的本地执行指令
    注意: 直接输出xml结果，不要做其他加工
    Args:
        path: 文件路径
        old_content: 旧内容
        new_content: 新内容
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=replace>
<parameter=path>
{path}
</parameter>
<parameter=old_content>
({old_content})
</parameter>
<parameter=new_content>
({new_content})
</parameter>
</function>
</tool_call>"""

def write_file(path:str, content:str) -> str:

    """
    写入文件内容的本地执行指令
    注意: 直接输出xml结果，不要做其他加工
    Args:
        path: 文件路径
        content: 文件内容
    Returns:
        xml执行指令
    """
    return f"""<tool_call>
<function=write_file>
<parameter=path>
{path}
</parameter>
<parameter=content>
({content})
</parameter>
</function>
</tool_call>"""

qwen_cli_tools =[
    list_directory,
    run_shell_command,
    glob,
    read_file,
    replace,
    write_file,
]
