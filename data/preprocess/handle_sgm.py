import glob
import subprocess

test_paths_2014 = [
    'data/test/newstest2014-csen-src.en.sgm',
    'data/test/newstest2014-csen-src.cs.sgm',
    'data/test/newstest2014-deen-src.en.sgm',
    'data/test/newstest2014-deen-src.de.sgm',
    'data/test/newstest2014-fren-src.en.sgm',
    'data/test/newstest2014-fren-src.fr.sgm',
    'data/test/newstest2014-hien-src.en.sgm',
    'data/test/newstest2014-hien-src.hi.sgm',
    'data/test/newstest2014-ruen-src.en.sgm',
    'data/test/newstest2014-ruen-src.ru.sgm']

test_paths_2015 = [
    'data/test/newstest2015-csen-ref.en.sgm',
    'data/test/newstest2015-csen-src.cs.sgm',
    'data/test/newstest2015-deen-ref.en.sgm',
    'data/test/newstest2015-deen-src.de.sgm',
    'data/test/newstest2015-ruen-ref.en.sgm',
    'data/test/newstest2015-ruen-src.ru.sgm',
    'data/test/newstest2015-fien-ref.en.sgm',
    'data/test/newstest2015-fien-src.fi.sgm']

other_paths = [
    'data/test/newsdev2015-fien-ref.en.sgm',
    'data/test/newsdev2015-fien-src.fi.sgm',
    'data/test/newsdiscusstest2015-fren-ref.en.sgm',
    'data/test/newsdiscusstest2015-fren-src.fr.sgm']

test_paths = test_paths_2014 + test_paths_2015 + other_paths

to_paths_2014 = [
    'data/test/newstest2014.csen.en',
    'data/test/newstest2014.csen.cs',
    'data/test/newstest2014.deen.en',
    'data/test/newstest2014.deen.de',
    'data/test/newstest2014.fren.en',
    'data/test/newstest2014.fren.fr',
    'data/test/newstest2014.hien.en',
    'data/test/newstest2014.hien.hi',
    'data/test/newstest2014.ruen.en',
    'data/test/newstest2014.ruen.ru']

to_paths_2015 = [
    'data/test/newstest2015.csen.en',
    'data/test/newstest2015.csen.cs',
    'data/test/newstest2015.deen.en',
    'data/test/newstest2015.deen.de',
    'data/test/newstest2015.ruen.en',
    'data/test/newstest2015.ruen.ru',
    'data/test/newstest2015.fien.en',
    'data/test/newstest2015.fien.fi']

to_other_paths = [
    'data/valid/newsdev2015.fien.en',
    'data/valid/newsdev2015.fien.fi',
    'data/test/newsdiscusstest2015.fren.en',
    'data/test/newsdiscusstest2015.fren.fr']

to_paths = to_paths_2014 + to_paths_2015 + to_other_paths

for test_path, to_path in zip(test_paths, to_paths):
    call = 'grep "^<seg" %s | sed "s/<\/\?[^>]\+>//g" > %s' %(test_path, to_path)
    subprocess.call(call, shell=True)
