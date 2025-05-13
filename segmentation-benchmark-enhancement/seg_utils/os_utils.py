import os


def is_pycharm_and_vscode_hosted():
    return (
        bool(os.getenv('PYCHARM_HOSTED', False))
        or (os.getenv('TERM_PROGRAM', False) == 'vscode')
    )
