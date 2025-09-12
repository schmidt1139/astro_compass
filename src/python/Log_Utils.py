def log(info, log, flag_report_to_console=False):

    log.append(info)

    if flag_report_to_console:
        print(info)

    return log
