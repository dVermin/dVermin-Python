import os
import stat
import subprocess
from id_writer import write_id_to_project
from attr_writer import write_attr_to_project
import shutil
import glob
import re
import logging.handlers
import datetime

build_tool_dir = "Android\\Sdk\\build-tools\\31.0.0"
keystore_path = "apk-helper.keystore"

zip_aligner_path = os.path.join(build_tool_dir, "zipalign.exe")
apk_signer_path = os.path.join(build_tool_dir, "lib", "apksigner.jar")


logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)

rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

f_handler = logging.FileHandler('error.log')
f_handler.setLevel(logging.ERROR)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)
logger.addHandler(f_handler)


def make_legitmate_path(original_path):
    if " " in original_path:
        original_path = '"' + original_path + '"'
    return original_path


def decompile(filename, output_dir):
    filename = make_legitmate_path(filename)
    output_dir = make_legitmate_path(output_dir)
    status = execute_command(['apktool.bat', 'd',  filename, '-f', '-o', output_dir])
    if status:
        return True



def rebuild(out_dir):
    out_dir = make_legitmate_path(out_dir)
    status = execute_command(['apktool.bat', 'b', out_dir])
    if status:
        return True
    status = execute_command(['apktool.bat', 'b', out_dir])
    if status:
        return True


def zip_aligner(out_dir, apk):
    apk_path = os.path.join(out_dir, "dist", apk)
    apk_path = make_legitmate_path(apk_path)
    new_apk_path = os.path.join(out_dir, "dist", "out.apk")
    new_apk_path = make_legitmate_path(new_apk_path)
    # status = execute_command([zip_aligner_path, '-v', '4', apk_path, new_apk_path])
    status = execute_command([zip_aligner_path,  '-p','-f', '4', apk_path, new_apk_path])
    if status:
        return True

def execute_command(command, communicate=None):
    failed = False
    apktool_command = " ".join(command)
    p = subprocess.Popen(apktool_command, stderr=subprocess.PIPE, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if communicate:
        stdout, stderr = p.communicate(communicate)
    else:
        stdout, stderr = p.communicate()
    if stderr is not None and stderr != b'':
        failed = True
    print("STDOUT:", stdout, "STDERR:", stderr, sep="\n")
    p.stdout.close()
    p.wait()
    if failed:
        logger.error(f"CMD: {apktool_command}\nSTDOUT: {stdout}\nSTDERR: {stderr}")
        return False
    else:
        logger.info(f"CMD: {apktool_command}\nSTDOUT: {stdout}\nSTDERR: {stderr}")
        return True


def apk_signer(apk_dir):
    new_apk_path = os.path.join(apk_dir, "dist", "out.apk")

    status = execute_command(['java', '-jar', '"'+apk_signer_path+'"', 'sign', '--ks', '"'+keystore_path+'"', '"'+new_apk_path+'"'], b'123456')
    if status:
        return True

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

if __name__ == "__main__":

    files = glob.glob('path/to/apk/*.apk')
    dataset_dir = os.path.join("..", "outs")
    files = sorted(files, key=lambda x:float(re.search("(^[0-9]+)",os.path.basename(x))[0])*-1)
    for file in files:
        apk_name = os.path.basename(file)
        if os.path.exists(os.path.join(dataset_dir, "out", apk_name)):
            continue
        smali_project_path = os.path.join(dataset_dir, os.path.splitext(apk_name)[0])
        # try:
        decompile(file, smali_project_path)
        write_id_to_project(smali_project_path)
        write_attr_to_project(smali_project_path)
        rebuild(smali_project_path)
        zip_aligner(smali_project_path, apk_name)
        apk_signer(smali_project_path)
        shutil.copy(os.path.join(smali_project_path, "dist", "out.apk"), os.path.join(dataset_dir, "out", apk_name))
        shutil.copytree(os.path.join(smali_project_path, "res", "layout"),
                    os.path.join(dataset_dir, "layout", os.path.splitext(apk_name)[0]), dirs_exist_ok=True)
        shutil.rmtree(smali_project_path, ignore_errors=False)



