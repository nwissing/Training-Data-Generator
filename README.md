# Training-Data-Generator

## Frontend of the application. 
The Index.html represents the frontend. The following components were defined there
- css styles: Specifies the appearance of the application including the layout, the buttons, or the design of the drawn prompts 
- html structure:
  - <em>Starting Page:</em> displays the Leaflet map and provides access to the Popup Windows and the QuickGuide.
  - <em>Popup Window:</em> 
  - <em>QuickGuide:</em> 
  - <em>Loading overlay:</em> is displayed after the algorithm for segmentation has been started.
- Button functionalities:
  - <em>quickGuideButton:</em>
  - <em>maskButton:</em>
  - <em>btn_segmented_image:</em>
  - <em>popupmap:</em> the map in the Popup Window is a button since it is used to display the prompts, when the left mouse button is pressed
  - <em>clearPrompts:</em>
  - <em>downloadButton1:</em> download the last segmented image as jpg (1 file: mask and image combined)
  - <em>downloadButton2:</em> download the last segmented image as npy (2 files: one for the mask, one for the image)
  - <em>closeButton:</em> closes the Popup Window
  - <em>closeButton2:</em> close QuickGuide window
  
