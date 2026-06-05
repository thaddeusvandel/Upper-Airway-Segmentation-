import json
import codecs

with codecs.open('SegMod.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with codecs.open('notebook_summary.txt', 'w', encoding='utf-8') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if any(keyword in source.lower() for keyword in ['unet', 'train', 'transform', 'loss', 'optimizer']):
                f.write(source)
                f.write('\n\n' + '='*80 + '\n\n')
