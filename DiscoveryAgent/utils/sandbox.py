import os
import shutil

def init_sandbox_dir(data_path, destination_path, claude_md_template_path = None, init_code_dir_list = None):

    # Create destination directory
    os.makedirs(destination_path, exist_ok=True)

    # Ensure data symlink is ignored by git in sandbox
    gitignore_path = os.path.join(destination_path, ".gitignore")
    try:
        with open(gitignore_path, "w") as gitignore_file:
            gitignore_file.write("data/*\n")
    except Exception as e:
        print(f"Warning: could not write .gitignore in sandbox: {e}")

    # Make a real "data" directory in destination_path and symlink its contents
    link_path = os.path.join(destination_path, "data")
    # Remove existing link/file if present
    if os.path.islink(link_path) or os.path.exists(link_path):
        if os.path.islink(link_path) or os.path.isfile(link_path):
            os.unlink(link_path)
        elif os.path.isdir(link_path):
            shutil.rmtree(link_path)
    os.makedirs(link_path, exist_ok=True) # Mkdir data directory

    # Symlink each top-level entry from data_path into the sandbox data dir
    for entry in os.listdir(data_path):
        source_entry = os.path.join(data_path, entry)
        dest_entry = os.path.join(link_path, entry)
        if os.path.islink(dest_entry) or os.path.exists(dest_entry):
            if os.path.islink(dest_entry) or os.path.isfile(dest_entry):
                os.unlink(dest_entry)
            elif os.path.isdir(dest_entry):
                shutil.rmtree(dest_entry)
        os.symlink(os.path.abspath(source_entry), dest_entry)

    if claude_md_template_path is not None:
        shutil.copy(claude_md_template_path, os.path.join(destination_path, "CLAUDE.md"))

    if init_code_dir_list:
        if isinstance(init_code_dir_list, str):
            init_code_dir_list = [
                entry.strip()
                for entry in init_code_dir_list.split(",")
                if entry.strip()
            ]
        for path_entry in init_code_dir_list:
            source_path = os.path.abspath(os.path.expanduser(path_entry))
            if not os.path.exists(source_path):
                print(f"Warning: init_code_dir_list path not found: {source_path}")
                continue
            dest_name = os.path.basename(source_path.rstrip(os.sep))
            destination = os.path.join(destination_path, dest_name)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, destination)

    return destination_path