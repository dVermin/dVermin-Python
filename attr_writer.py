from lxml import etree
import uuid
import os
import glob

import config


def write_attr_to_project(smali_dir):
    attr_file_path = os.path.join(smali_dir, 'res', 'values', 'attrs.xml')
    my_parser = etree.XMLParser(encoding="utf-8")

    if os.path.exists(attr_file_path):
        tree = etree.parse(attr_file_path, parser=my_parser)
        root = tree.getroot()
        query = root.xpath("//attr[@format='string' and @name='{0}']".
                           format(config.injected_id_name))
        if len(query) == 0:
            attr_node = etree.Element('attr')
            attr_node.attrib["name"] = config.injected_id_name
            attr_node.attrib["format"] = "string"
            root.append(attr_node)
            tree.write(attr_file_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
    else:
        raise RuntimeError('There is no file named attrs.xml in smali project')

