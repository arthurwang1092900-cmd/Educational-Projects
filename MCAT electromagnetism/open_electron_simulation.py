"""
open_electron_simulation.py

Opens the electron simulation (electron_simulation.html) in your web browser.
The simulation itself lives in the HTML file, so it runs fast and has
clickable buttons and sliders.

How to run (Anaconda Prompt):
    cd Downloads
    python open_electron_simulation.py

Keep this file and electron_simulation.html in the SAME folder.

New thing (not in the course modules):
  - webbrowser.open() : opens a web page in your browser
"""

import os
import webbrowser

# the folder this script is in
folder = os.path.dirname(os.path.abspath(__file__))

# the full path of the html file
html_file = folder + "\\electron_simulation.html"

# check the file is really there
if os.path.exists(html_file):
    print("Opening " + html_file)
    webbrowser.open("file:///" + html_file.replace("\\", "/"))
else:
    print("Could not find electron_simulation.html in " + folder)
    print("Put both files in the same folder and try again.")
