import gdown

url = 'https://drive.google.com/drive/folders/1GV3F8oqn3N-hfuHpAigjBRHYBtEIH5xR'
output = 'data'
gdown.download(url, output, quiet=False)