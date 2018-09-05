# fontのあるディレクトリで実行するとcssを生成する

import sys,os

files = os.listdir("./")
css = ""

for f in files:
    t = '@font-face{{\n' \
    'font-family : "{0}";\n' \
    'src : url("./font100/{0}");\n' \
    '}}\n'
    # print(t)
    t = t.format(f)

    css = css + t

with open('./font100.css','w') as fp:
    fp.write(css)
