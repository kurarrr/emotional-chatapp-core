with open('./font16_valence.txt','r') as f:
    s = f.read()

dat = []
dat_re = []
lines = s.split('\n')
print(len(lines))

for line in lines:
    ary = line.split("\t")
    if len(ary) != 2:
        print(ary)
        continue

    ext = '.ttf'
    

    dat_re.append({
        'name' : ary[0] + ext,
        'Valence' : float(ary[1]),
        'Arousal' : 0.0
    })

import json

with open('./font16_valence_re.json','w') as f:
    json.dump(dat_re,f)

# otfは手動で直す 