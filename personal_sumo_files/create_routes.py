import xml.etree.ElementTree as ElementTree
from lxml import etree

def create_routes():

    parser = etree.XMLParser(recover=True)
    tree = ElementTree.parse("lemgo_small2.rou.xml", parser=parser)
    root = tree.getroot()

    for vehicle in root.findall("vehicle"):
        for route in vehicle.findall("route"):
            edges = route.attrib["edges"]
            first_edge = edges.split(" ")[0]
        
        root.remove(vehicle)
        root.append(etree.Element("route"))
        route = root[-1]
        route.set('edges', edges)
        route.set('id', first_edge)

    tree.write("lemgo_small2_out.rou.xml")


if __name__ == "__main__":
    create_routes()