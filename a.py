import re
with open('requirements.txt') as fd:
    lines = fd.readlines()
    newlines= ''
    for line in lines:
        list = re.split(r"\s+",line)
        op = list[0]+'=='+list[1]
        newlines += op + '\n'
    print(newlines)