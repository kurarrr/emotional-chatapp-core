with open('./font100_vad.txt','r') as f:
    s = f.read()

dat = []
dat_re = []
lines = s.split('\n')
print(len(lines))

oft_list = ['4420','4447']

for line in lines:
    ary = line.split("\t")
    if len(ary) != 3:
        print(ary)
        continue

    ext = '.ttf'
    if ary[0] in oft_list:
        ext = '.otf'
    

    dat.append({
        'name' : ary[0] + ext,
        'Valence' : ary[1],
        'Arousal' : ary[2]
    })
    dat_re.append({
        'name' : ary[0] + ext,
        'Valence' : (float(ary[1])-5.0)/4,
        'Arousal' : (float(ary[2])-5.0)/4
    })

import json

with open('./font100_vad.json','w') as f:
    json.dump(dat,f)

with open('./font100_vad_re.json','w') as f:
    json.dump(dat_re,f)

# otfは手動で直す 