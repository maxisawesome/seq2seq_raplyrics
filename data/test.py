import pathlib
for l in open(pathlib.Path(__file__).parents[0].resolve() / 'words_in_lyrics'):
    print(l)
