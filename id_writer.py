from lxml import etree
import uuid
import os
import glob

import config

content_description_keys = [
    "navigationContentDescription",
    'actionModeCloseContentDescription',
    "collapseContentDescription"
]


def set_uuid_to_description(tags, id_uuid):
    mapping = tags.nsmap
    for content_description_key in content_description_keys:
        if 'app' in mapping:
            attrib_key_android = "{" + mapping['app'] + "}" + content_description_key
            if attrib_key_android in tags.attrib:
                tags.attrib[attrib_key_android] = id_uuid


def write_id_to_project(smali_dir):
    layout_dir = os.path.join(smali_dir, 'res', 'layout*', '*.xml')
    NS = "http://schemas.android.com/apk/res-auto"
    myparser = etree.XMLParser(encoding="utf-8")

    for layout_file in glob.glob(layout_dir):
        tree = etree.parse(layout_file, parser=myparser)

        root = tree.getroot()
        nsmap = root.nsmap

        nsmap['helperNS'] = NS
        new_root = etree.Element(root.tag, nsmap=nsmap)
        for sub_attrib in root.attrib:
            new_root.attrib[sub_attrib] = root.attrib[sub_attrib]

        new_root[:] = root[:]


        for tags in new_root.iter():
            mapping = tags.nsmap
            attrib_key_helperNS = "{" + mapping['helperNS'] + "}"+config.injected_id_name
            id_uuid = uuid.uuid4().hex[:6]
            tags.attrib[attrib_key_helperNS] = id_uuid

            if "android" in mapping:
                attrib_key_android = "{" + mapping['android'] + "}contentDescription"
                tags.attrib[attrib_key_android] = id_uuid
            set_uuid_to_description(tags, id_uuid)

        new_root.getroottree().write(layout_file, pretty_print=True, xml_declaration=True, encoding='utf-8')