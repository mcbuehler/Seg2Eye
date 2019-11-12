import errno
import os
import shutil


def listdir(path, prefix='', postfix='', return_prefix=True,
            return_postfix=True):
    """
    Lists all files in path that start with prefix and end with postfix.
    By default, this function returns all filenames. If you do not want to
    return the pre- or postfix, set the corresponding parameters to False.
    :param path:
    :param prefix:
    :param postfix:
    :param return_prefix:
    :param return_postfix:
    :return: list(str)
    """
    files = os.listdir(path)
    filtered_files = filter(
        lambda f: f.startswith(prefix) and f.endswith(postfix), files)
    return_files = filtered_files
    if not return_prefix:
        idx_start = len(prefix) - 1
        return_files = [f[idx_start:] for f in filtered_files]
    if not return_postfix:
        idx_end = len(postfix) - 1
        return_files = [f[:-idx_end] for f in filtered_files]
    return_files = set(return_files)
    return list(return_files)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def copy(src, dest, overwrite=False):
    if os.path.exists(dest) and overwrite:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        else:
            os.remove(dest)
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def copy_src(path_from, path_to):
    # Collect all files and folders that contain python files
    python_files = listdir(path_from, postfix='.py')
    to_save = python_files
    folders = listdir(path_from)
    for f in folders:
        if os.path.isdir(f) and '.py' in " ".join(listdir(f)):
            to_save.append(f)

    tmp_folder = os.path.join(path_to, 'src/')
    create_folder_if_not_exists(tmp_folder)
    for f in to_save:
        copy(f, os.path.join(tmp_folder, f), overwrite=True)
    shutil.make_archive(tmp_folder, 'zip', tmp_folder)
    try:
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        # We got a FileNotfound error on the cluster. Maybe some race conditions?
        pass
    print(f"Copied {len(to_save)} files and folders from {path_from} to {path_to}")


