
"""
Turned the monolithic XML file into a split files with its 
top-level elements one-per-line so they can sensibly be 
approached with Spark.

Precondition:
    - download the 'xxx.osm.bz2' file from https://planet.openstreetmap.org/

Typical invocation (Working directory: ./datasets/openstreetmap_north):
pv ./north-america-latest.osm.bz2 | bzcat | ../src/osm_01_disassemble.py | split -C1G -d -a 4 --additional-suffix='.xml' --filter='gzip > $FILE.gz' - osm-north-america-

Reference: https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/ProjectTour
"""


import sys
from lxml import etree


def main(instream, outstream):
    # based on https://stackoverflow.com/a/35309644
    parser = etree.iterparse(instream, events = ['start', 'end'], remove_blank_text = True)    
    _, root = next(parser) # consume root element
    record_nesting_level = 0
    root.clear()
    del root

    for event, elem in parser:
        if event == 'start': 
            record_nesting_level += 1
        elif event == 'end':
            record_nesting_level -= 1
            if record_nesting_level != 0:
                continue

            xml = etree.tostring(elem).strip()
            assert b'\n' not in xml
            outstream.write(xml)
            outstream.write(b'\n')

            # don't leak memory, per https://stackoverflow.com/a/9814580
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

main(sys.stdin.buffer, sys.stdout.buffer)

