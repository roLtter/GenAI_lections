import json
import xml.etree.ElementTree as ET


def main():
    root = ET.parse("coverage.xml").getroot()

    line_rate = float(root.attrib["line-rate"])
    coverage = round(line_rate * 100, 2)

    data = {
        "schemaVersion": 1,
        "label": "My Code Coverage",
        "message": f"{coverage}%",
        "color": (
            "brightgreen"
            if coverage >= 80
            else "yellow"
            if coverage >= 50
            else "red"
        ),
    }

    with open("coverage_badge.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
