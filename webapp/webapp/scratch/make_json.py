with open('./font100_vad.txt','r') as f:
    s = f.read()

dat = []
lines = s.split('\n')
print(len(lines))

for line in lines:
    ary = line.split("\t")
    if len(ary) != 3:
        print(ary)
        continue
    dat.append({
        'name' : ary[0] + '.ttf',
        'Valence' : ary[1],
        'Arousal' : ary[2]
    })

import json

with open('./font100_vad.json','w') as f:
    json.dump(dat,f)

# otfは手動で直す