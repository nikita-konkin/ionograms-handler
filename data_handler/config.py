from configparser import ConfigParser
import os 



def config(filename='database2.ini', section='db'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + '\\'+filename
    parser = ConfigParser()
    parser.read(filename)
    # print(filename)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            # print(param)
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def day_num():
    filename='adress_python.ini'
    section='day_num'
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    dn = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            # print(param)
            dn[param[0]] = param[1]
        # print(dn)
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return dn

# if __name__ == "__main__":
#     print(day_num()['day_start'])
