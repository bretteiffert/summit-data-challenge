#!/usr/bin/env python3

"""
Converting summit login node log files into dataframes, and save the pruned
dataframes into compressed feather files for subsequent analysis.
"""

import os
import glob
import io
import subprocess
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
from datetime import timedelta

from tqdm.notebook import tqdm
from datetime import datetime

import logging

logging.basicConfig(format="[%(levelname)s]%(asctime)s:%(message)s", level=logging.INFO)


def grep_location(pattern: str, filename: str) -> list:
    """
    Find line numbers of the provided pattern in a given file

    Parameter
    ---------
    @param pattern:
        regular expression pattern needed by GNU grep
    @param filename:
        full path to the log file

    Returns
    -------
    @returns
        list of line numbers (0-based) that matches the pattern.
    """
    cmd = f"""grep -n "{pattern}" {filename}"""
    rst = subprocess.run(cmd, capture_output=True, shell=True)
    stdout = rst.stdout.decode("utf-8").strip()
    # the -2 should be the location
    locs = [int(line.split(":")[-2]) - 1 for line in stdout.split("\n")]
    return locs


def read_logfile(filename: str) -> list:
    """
    reader that handles file encoding issues

    Parameter
    ---------
    @param filename:
        full path to log file

    Returns
    -------
    @returns:
        list of lines
    """
    # try opening with default encoding (utf-8)
    try:
        with open(filename, "r", encoding="utf-8") as fn:
            lines = fn.readlines()
    except UnicodeDecodeError:
        logging.warning(f"Cannot open {filename} with utf-8, switch to latin-1")
        with open(filename, "r", encoding="latin-1") as fn:
            lines = fn.readlines()
    #
    return lines


def get_date_str(filename: str) -> str:
    """
    Get the date string from given log file.

    Parameter
    ---------
    @param filename:
        full path to log file

    Return
    ------
    @returns:
        date string

    NOTE
    ----
    modify this func to extend for support of different log file name patterns.
    """
    basename = os.path.basename(filename)
    # currently all log files are following the pattern
    # login[NODE_ID].summit.olcf.ornl.gov.[Month_Abbre][Day]_[Year].txt
    return basename.split(".")[-2]


def get_login_node_id(filename: str) -> int:
    """
    Get the login node ID from given log file name.

    Parameter
    ---------
    @param filename:
        full path to log file

    Return
    ------
    @returns:
        login node id
    """
    basename = os.path.basename(filename)
    # currently all log files are following the pattern
    # login[NODE_ID].summit.olcf.ornl.gov.[Month_Abbre][Day]_[Year].txt
    return int(basename.split(".")[0].replace("login", ""))


def extract_w_block(filename: str) -> pd.DataFrame:
    """
    Parse the w block in the given logfile into a DataFrame.

    Parameter
    ---------
    @param filename:
        full path to log file

    Returns
    -------
    @returns:
        dataframe containing the information from the w block.
    """
    # read into memory
    flines = read_logfile(filename)
    # get the date string
    date_str = get_date_str(filename)
    # process
    start_pattern = "^w --$"
    end_pattern = "^endw --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    df_list = []
    header = ["USER", "TTY", "FROM", "LOGIN@", "IDLE", "JCPU", "PCPU", "WHAT"]
    for i, locs in enumerate(zip(locs_start, locs_end)):
        idx_start = locs[0] + 3
        idx_end = locs[1]
        # build the string
        entries = [
            line.split()[:7] + [" ".join(line.split()[7:])]
            for line in flines[idx_start:idx_end]
        ]
        # buile the dataframe
        df = pd.DataFrame(entries, columns=header)
        # add summary info
        rawstring = flines[locs[0] + 1]
        # -- average load info
        avarege_load = rawstring.split("load average:")[-1].split(",")
        df["avg_load_over_1min"] = float(avarege_load[0])
        df["avg_load_over_5min"] = float(avarege_load[1])
        df["avg_load_over_15min"] = float(avarege_load[2])
        # -- number of users
        df["num_users"] = int(
            rawstring.split("load average:")[0].split(",")[-2].split()[0]
        )
        # -- add timestamp
        time_str = rawstring.split()[0]
        current_tiem = datetime.strptime(f"{date_str} {time_str}", "%b%d_%Y %H:%M:%S")
        df["time"] = current_tiem
        # -- add hour of the day
        # NOTE: could be useful for merging dataframes later
        df["hour"] = i
        #
        df_list.append(df)
    # return the df
    df = pd.concat(df_list)
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df

def extract_meminfo_block(filename, flines=None):
    # load file to memory
    flines = read_logfile(filename)
    # get date_string
    date_str = get_date_str(filename)
    current_date = datetime.strptime(date_str, "%b%d_%Y")
    # process
    start_pattern = "^meminfo --$"
    end_pattern = "^endmeminfo --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    data_list = []
    header = None
    #
    for i, locs in enumerate(zip(locs_start, locs_end)):
        rawlines = flines[locs[0]+1 : locs[1]]
        if not header:
            header = [f"{line.split(':')[0]} kB" for line in rawlines]
        vals = [int(line.split(":")[1].split()[0]) for line in rawlines] + [i]
        #
        data_list.append(vals)
    #
    df = pd.DataFrame(
        data_list,
        columns=header + ["hour"],
    )
    df["date"] = current_date
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df

def extract_vmstat_block(filename, flines=None):
    # load file to memory
    flines = read_logfile(filename)
    # get date_string
    date_str = get_date_str(filename)
    current_date = datetime.strptime(date_str, "%b%d_%Y")
    # process
    start_pattern = "^vmstat --$"
    end_pattern = "^endvmstat --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    data_list = []
    header = None
    #
    for i, locs in enumerate(zip(locs_start, locs_end)):
        rawlines = flines[locs[0]+1 : locs[1]]
        if not header:
            # TODO: need to know the unit here
            header = [f"{line.split()[0]}" for line in rawlines]
        vals = [int(line.split()[1].strip()) for line in rawlines] + [i]
        #
        data_list.append(vals)
    #
    df = pd.DataFrame(
        data_list,
        columns=header + ["hour"],
    )
    df["date"] = current_date
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df

def extract_ps_block(filename: str) -> pd.DataFrame:
    """
    Parse the ps block in the given logfile into a DataFrame.

    Parameter
    ---------
    @param filename:
        full path to log file

    Returns
    -------
    @returns:
        dataframe containing the information from the ps block.
    """
    # read into memory
    flines = read_logfile(filename)
    # get the date string
    date_str = get_date_str(filename)
    current_date = datetime.strptime(date_str, "%b%d_%Y")
    # process
    start_pattern = "^ps aux --$"
    end_pattern = "^endps aux --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    df_list = []
    header = [
        "USER",
        "PID",
        "%CPU",
        "%MEM",
        "VSZ",
        "RSS",
        "TTY",
        "STAT",
        "START",
        "TIME",
        "COMMAND",
    ]
    for i, locs in enumerate(zip(locs_start, locs_end)):
        idx_start = locs[0] + 2
        idx_end = locs[1]
        # build the string
        entries = [
            line.split()[:10] + [" ".join(line.split()[10:])]
            for line in flines[idx_start:idx_end]
        ]
        # buile the dataframe
        df = pd.DataFrame(entries, columns=header)
        # -- add hour of the day
        # NOTE: could be useful for merging dataframes later
        df["hour"] = i
        #
        df["date"] = current_date
        #
        df_list.append(df)
    # return the df
    df = pd.concat(df_list)
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df


def extract_top_block(filename: str) -> pd.DataFrame:
    """
    Parse the top block in the given logfile into a DataFrame.

    Parameter
    ---------
    @param filename:
        full path to log file

    Returns
    -------
    @returns:
        dataframe containing the information from the top block.
    """
    # read into memory
    flines = read_logfile(filename)
    # get the date string
    date_str = get_date_str(filename)
    # process
    start_pattern = "^top -n 1 -bc | awk"
    end_pattern = "^endtop -n 1 -bc | awk"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    df_list = []
    header = [
        "PID",
        "USER",
        "PR",
        "NI",
        "VIRT",
        "RES",
        "SHR",
        "S",
        "%CPU",
        "%MEM",
        "TIME+",
        "COMMAND",
    ]
    for i, locs in enumerate(zip(locs_start, locs_end)):
        idx_start = locs[0] + 8
        idx_end = locs[1]
        # build the string
        entries = [
            line.split()[:11] + [" ".join(line.split()[11:])]
            for line in flines[idx_start:idx_end]
        ]
        # buile the dataframe
        df = pd.DataFrame(entries, columns=header)
        # add summary info
        # - first row
        rawstring = flines[locs[0] + 1]
        # -- average load info
        avarege_load = rawstring.split("load average:")[-1].split(",")
        df["avg_load_over_1min"] = float(avarege_load[0])
        df["avg_load_over_5min"] = float(avarege_load[1])
        df["avg_load_over_15min"] = float(avarege_load[2])
        # -- number of users
        df["num_users"] = int(
            rawstring.split("load average:")[0].split(",")[-2].split()[0]
        )
        # -- add timestamp
        time_str = rawstring.split()[2]
        current_tiem = datetime.strptime(f"{date_str} {time_str}", "%b%d_%Y %H:%M:%S")
        df["time"] = current_tiem
        # -- add hour of the day
        # NOTE: could be useful for merging dataframes later
        df["hour"] = i
        # - task row
        rawstring = flines[locs[0] + 2]
        vals = re.findall("\d+", rawstring)
        labels = [
            "task_total",
            "task_runing",
            "task_sleeping",
            "task_stopped",
            "task_zombie",
        ]
        for label, val in zip(labels, vals):
            df[label] = val
        # - cpu row
        rawstring = flines[locs[0] + 3]
        vals = re.findall("\d+.\d+", rawstring)
        labels = [
            "cpu_us",
            "cpu_sy",
            "cpu_ni",
            "cpu_id",
            "cpu_wa",
            "cpu_hi",
            "cpu_si",
            "cpu_st",
        ]
        for label, val in zip(labels, vals):
            df[label] = val
        # - mem row
        rawstring = flines[locs[0] + 4]
        vals = re.findall("\d+.\d+", rawstring)
        labels = [
            "mem_total_KiB",
            "mem_free_KiB",
            "mem_used_KiB",
            "mem_buff_KiB",
        ]
        for label, val in zip(labels, vals):
            df[label] = val
        # - swap row
        rawstring = flines[locs[0] + 5]
        vals = re.findall("\d+.\d+", rawstring)
        labels = [
            "swap_total_KiB",
            "swap_free_KiB",
            "swap_used_KiB",
            "swap_avail_mem_KiB",
        ]
        for label, val in zip(labels, vals):
            df[label] = val
        #
        df_list.append(df)
    # return the df
    df = pd.concat(df_list)
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df


def extract_bjobs_block(filename: str) -> pd.DataFrame:
    """
    Parse the bjobs block in the given logfile into a DataFrame.

    Parameter
    ---------
    @param filename:
        full path to log file

    Returns
    -------
    @returns:
        dataframe containing the information from the bjobs block.
    """
    # read into memory
    flines = read_logfile(filename)
    # get the date string
    date_str = get_date_str(filename)
    current_date = datetime.strptime(date_str, "%b%d_%Y")
    # datetime needs year to deal with leap day
    current_year = date_str.split("_")[-1]
    # process
    start_pattern = "^bjobs -a -u all --$"
    end_pattern = "^endbjobs -a -u all --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    entry_list = []
    header = [
        "JOBID",
        "USER",
        "STAT",
        "SLOTS",
        "QUEUE",
        "START_TIME",
        "FINISH_TIME",
        "JOB_NAME",
        "hour",
    ]
    for i, locs in enumerate(zip(locs_start, locs_end)):
        idx_start = locs[0] + 2
        idx_end = locs[1]
        for line in flines[idx_start:idx_end]:
            vals_list = line.split()
            if len(vals_list) == len(header) - 1:
                entry_list.append(vals_list + [i])
            else:
                # have acutal datetime inside, need parsing
                tmp = vals_list[:5]
                start_time = pd.NaT
                end_time = pd.NaT
                if vals_list[5] == "-":
                    if vals_list[6] == "-":
                        pass  # note, this branch should already handle with the top level
                    else:
                        end_time_str = " ".join(vals_list[6:9])
                        end_time_str = f"{current_year} {end_time_str}"
                        end_time = datetime.strptime(end_time_str, "%Y %b %d %H:%M")
                else:
                    start_time_str = " ".join(vals_list[5:8])
                    start_time_str = f"{current_year} {start_time_str}"
                    start_time = datetime.strptime(start_time_str, "%Y %b %d %H:%M")
                    if vals_list[8] == "-":
                        pass
                    else:
                        end_time_str = " ".join(vals_list[8:11])
                        end_time_str = f"{current_year} {end_time_str}"
                        end_time = datetime.strptime(end_time_str, "%Y %b %d %H:%M")
                #
                tmp += [start_time, end_time, vals_list[-1], i]
                #
                entry_list.append(tmp)
    #
    df = pd.DataFrame(entry_list, columns=header)
    df["date"] = current_date
    df["login_node"] = get_login_node_id(filename)
    df["START_TIME"] = df["START_TIME"].astype(str)
    df["FINISH_TIME"] = df["FINISH_TIME"].astype(str)
    return df

def parse_time(time_str):
    """
    Parse a time string e.g. (2h13m) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 2h13m)
    :return datetime.timedelta: A datetime.timedelta object
    """
    try:
        regex = re.compile(r'^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)?((?P<seconds>[\.\d]+?)s)?$')
        parts = regex.match(time_str)
        assert parts is not None, "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(time_str)
        time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
        return timedelta(**time_params)
    except:
        return pd.NaT

def extract_io_block(filename, flines=None):
    # load file to memory
    flines = read_logfile(filename)
    # get date_string
    date_str = get_date_str(filename)
    current_date = datetime.strptime(date_str, "%b%d_%Y")
    # process
    start_pattern = "^home response time unaliased ls --$"
    end_pattern = "^endgpfs scratch response time to create a 1 G file --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    entry_list = []
    header = [
        "home_ls_real", "home_ls_user", "home_ls_sys",
        "home_color_ls_real", "home_color_ls_user", "home_color_ls_sys",
        "gpfs_1gfile_create_real", "gpfs_1gfile_create_user", "gpfs_1gfile_create_sys",
        "hour"
    ]
    for i, locs in enumerate(zip(locs_start, locs_end)):
        idx_start = locs[0]
        idx_end = locs[1]
        tmp = []
        # -- home_ls
        for j in range(3):
            tmp.append(parse_time(flines[idx_start+2+j].split()[-1]))
        # -- home_color_ls
        for j in range(3):
            tmp.append(parse_time(flines[idx_start+8+j].split()[-1]))
        # -- gpfs 1g file create
        for j in range(3):
            tmp.append(parse_time(flines[idx_start+14+j].split()[-1]))
        #
        entry_list.append(tmp + [i])
    #
    df = pd.DataFrame(entry_list, columns=header)
    df["date"] = current_date
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df

def extract_filesys_block(filename, flines=None):
    # load file to memory
    flines = read_logfile(filename)
    # get date_string
    date_str = get_date_str(filename)
    current_date = datetime.strptime(date_str, "%b%d_%Y")
    # process
    start_pattern = "^df --$"
    end_pattern = "^enddf --$"
    locs_start = grep_location(start_pattern, filename)
    locs_end = grep_location(end_pattern, filename)
    df_list = []
    for i, locs in enumerate(zip(locs_start, locs_end)):
        idx_start = locs[0]+1
        idx_end = locs[1]
        df = pd.read_csv(io.StringIO("\n".join(flines[idx_start:idx_end])), delim_whitespace=True)
        #
        df["hour"] = i
        #
        df.drop(["on"], axis=1, inplace=True)
        #
        df_list.append(df)
    #
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    df["date"] = current_date
    df["login_node"] = get_login_node_id(filename)
    df.reset_index(inplace=True, drop=True)
    return df