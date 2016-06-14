import glob
import subprocess

test_paths = [
    'data/test/newstest2014-csen-src.en.sgm',
    'data/test/newstest2014-csen-src.cs.sgm',
    'data/test/newstest2014-deen-src.de.sgm',
    'data/test/newstest2014-fren-src.fr.sgm',
    'data/test/newstest2014-hien-src.hi.sgm',
    'data/test/newstest2014-ruen-src.ru.sgm',
    'data/test/newstest2015-encs-src.en.sgm',
    'data/test/newstest2015-csen-src.cs.sgm',
    'data/test/newstest2015-deen-src.de.sgm',
    'data/test/newstest2015-ruen-src.ru.sgm',
    'data/test/newstest2015-fien-src.fi.sgm',
    'data/test/newsdev2015-enfi-src.en.sgm',
    'data/test/newsdev2015-fien-src.fi.sgm',
    'data/test/newsdiscusstest2015-enfr-src.en.sgm',
    'data/test/newsdiscusstest2015-fren-src.fr.sgm']

to_paths = [
    'data/test/newstest2014.en',
    'data/test/newstest2014.cs',
    'data/test/newstest2014.de',
    'data/test/newstest2014.fr',
    'data/test/newstest2014.hi',
    'data/test/newstest2014.ru',
    'data/test/newstest2015.en',
    'data/test/newstest2015.cs',
    'data/test/newstest2015.de',
    'data/test/newstest2015.ru',
    'data/test/newstest2015.fi',
    'data/valid/newsdev2015.en',
    'data/valid/newsdev2015.fi',
    'data/test/newsdiscusstest2015.en',
    'data/test/newsdiscusstest2015.fr']

for test_path, to_path in zip(test_paths, to_paths):
    call = 'grep "^<seg" %s | sed "s/<\/\?[^>]\+>//g" > %s' %(test_path, to_path)
    subprocess.call(call, shell=True)
