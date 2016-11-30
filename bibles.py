import codecs
import re
import numpy


class NIV(object):
    '''
    Use only Class attributes and Class methods sdfsdfsf sdfsdfsdfsdfsd
    sdfsdf sdfsdf sdfdssdfsdfsdfsdfsdf
    to achieve Singleton behaviour
    '''

    base = None
    old = None
    new = None
    chap = None
    chap_headlines = None
    old_indices = None
    new_indices = None

    @classmethod
    def read_in(cls,
                path_to_data='data/NIV.txt',
                encoding='utf-16'):

        f = codecs.open(path_to_data, encoding=encoding)
        base = f.read()

        # cut beginning
        main_part_rx = re.compile(r'Genesis.+?(Genesis.*)', re.DOTALL)
        base_no_beginning = re.findall(main_part_rx, base)[0]

        main_part_rx = re.compile(r'(Genesis.+?)Genesis.*', re.DOTALL)
        chap_headlines = re.findall(main_part_rx, base)[0]

        base = base_no_beginning

        # get chapter index
        chap_headlines = (re.sub('\r\n', '\n', chap_headlines)
                            .strip()
                            .split('\n'))

        chap_headlines = numpy.array(chap_headlines)
        # get rid of 'New Testament' headline
        chap_headlines = chap_headlines[chap_headlines != 'New Testament']

        # insert space after numbers
        base = re.sub(r'(\d+)([^st|nd|rd|th])', r'\1 \2', base)

        # get chapters, distinguished by two new lines
        chapters_rx = re.compile(r'[\r\n]{0,}(.+?)[\r\n]{3}', re.DOTALL)
        chapters = re.findall(chapters_rx, base)
        # prefer list over dict
        # chapters = dict(zip(chapters[0::2], chapters[1::2]))
        chapters = numpy.array(chapters[1::2])

        # create new and old testament:
        old_indices = range(0, 39)
        new_indices = range(39, len(chap_headlines))

        old = '[\r\n]{3}'.join(chapters[old_indices])
        new = '[\r\n]{3}'.join(chapters[new_indices])

        NIV.base = base
        NIV.old = old
        NIV.new = new
        NIV.chap = chapters
        NIV.chap_head = chap_headlines
        NIV.old_indices = old_indices
        NIV.new_indices = new_indices


if __name__ == '__main__':
    NIV.read_in()
