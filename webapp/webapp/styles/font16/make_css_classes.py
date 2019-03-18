# fontのあるディレクトリで実行するとcssを生成する

import sys,os

files = os.listdir("./")
css = ""

for f in files:
    t = """
    .font-{0}{{
        font-family : '{1}';
    }}
    """.format(f.replace('.','-'),f)
    # class名に.は使えないのでreplace

    css = css + t

with open('../font16-effect.css','w') as fp:
    fp.write(css)
