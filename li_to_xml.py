import io
import os
import re
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import tostring
from lxml import etree
import glob


def get_properties_from_string(line):
    class_pattern = "( *?)([\S]+?)@([\S]+?) "
    class_prog = re.compile(class_pattern)
    pattern = "([\S]+?)=([0-9]+?),"
    prog = re.compile(pattern)
    groups_iter = prog.finditer(line)
    property_dict = {}
    for result in class_prog.finditer(line):
        depth_span = result.span(1)
        class_span = result.span(2)
        id_span = result.span(3)
        depth = len(line[depth_span[0]:depth_span[1]])
        class_name = line[class_span[0]:class_span[1]]
        id_name = line[id_span[0]:id_span[1]]
        property_dict["class"] = class_name
        property_dict["depth"] = depth
        property_dict["id"] = id_name
        break

    for result in groups_iter:
        property_span = result.span(1)
        value_length_span = result.span(2)

        property = line[property_span[0]:property_span[1]]
        property = re.sub(r'[\(\)]', '', property)
        property = re.sub(r'[^A-z]', '_', property)
        value_length = int(line[value_length_span[0]:value_length_span[1]])
        value = line[value_length_span[1] + 1:value_length_span[1] + 1 + value_length]
        property_dict[property] = value
    return property_dict


def dict_to_xml(tag, d):
    """
    Turn a simple dict of key/value pairs into XML
    """
    elem = Element(tag)
    for key, val in d.items():
        elem.set(key, str(val))
    return elem


def write_xml(li_path, li_to_text_jar_path):
    li_new_xml_path = os.path.join(os.path.dirname(li_path), "ViewHierarchy.xml")

    if not os.path.exists(li_new_xml_path):

        li_txt_path = li_path.replace(".li", ".txt")
        command = ' '.join(['java', "-jar", li_to_text_jar_path, li_path, li_txt_path])
        os.system(command)

        with io.open(li_txt_path, 'r', encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
            parents = []
            for line in lines:
                if line == "DONE.":
                    break
                property_dict = get_properties_from_string(line)
                element_new = dict_to_xml("node", property_dict)
                depth = property_dict["depth"]
                if depth == 0:
                    parents.append(element_new)
                    continue
                else:
                    parents[depth - 1].append(element_new)
                    if depth > len(parents) - 1:
                        parents.append(element_new)
                    else:
                        parents[depth] = element_new
            xml_string = tostring(parents[0], encoding="utf-8").decode("utf-8")
            x = etree.fromstring(xml_string)
            with io.open(li_new_xml_path, "w", encoding="utf-8") as file_writer:
                file_writer.write(etree.tostring(x, encoding="utf-8", pretty_print=True).decode("utf-8"))
