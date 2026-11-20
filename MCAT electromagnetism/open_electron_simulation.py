"""
open_electron_simulation.py

Opens the electron simulation (electron_simulation.html) in Microsoft Edge.
The simulation itself lives in the HTML file, so it runs fast and has
clickable buttons and sliders.

How to run (Anaconda Prompt):
    cd Downloads
    python open_electron_simulation.py

Keep this file and electron_simulation.html in the SAME folder.

If you prefer Chrome, change BROWSER below from "msedge" to "chrome".

New thing (not in the course modules):
  - os.system() : runs a normal Windows command from Python
"""

import os

# which browser to open: "msedge" (Edge) or "chrome"
BROWSER = "msedge"

# the folder this script is in
folder = os.path.dirname(os.path.abspath(__file__))

# the full path of the html file
html_file = folder + "\\electron_simulation.html"

# check the file is really there
if os.path.exists(html_file):
    print("Opening " + html_file + " in " + BROWSER)
    # the Windows "start" command opens the file in the chosen browser
    os.system('start ' + BROWSER + ' "' + html_file + '"')
else:
    print("Could not find electron_simulation.html in " + folder)
    print("Put both files in the same folder and try again.")
