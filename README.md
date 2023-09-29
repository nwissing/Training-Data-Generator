# Training-Data-Generator

## Frontend of the application. 
The Index.html represents the frontend. The following components were defined there
- css styles: Specifies the appearance of the application including the layout, the buttons, or the design of the drawn prompts 
- html structure:
  - <em>Starting Page:</em> displays the Leaflet map and provides access to the Popup Windows and the QuickGuide.
  - <em>Popup Window:</em> presents the image to be segmented, the prompt selection and the download buttons
  - <em>QuickGuide:</em> shows a short tutorial on how to use the web application
  - <em>Loading overlay:</em> is displayed after the algorithm for segmentation has been started.
- Button functionalities:
  - <em>quickGuideButton:</em> opens the QuickGuide window
  - <em>maskButton:</em> opens the Popup Window. Displays the image that will be segmented.
  - <em>btn_segmented_image:</em> starts the segmentation by making an API request to the backend and passing the prompts and the image as parameters
  - <em>popupmap:</em> the map in the Popup Window is a button since it is used to display the prompts, when the left mouse button is pressed
  - <em>clearPrompts:</em> deletes the previous prompts on the image
  - <em>downloadButton1:</em> download the last segmented image as jpg (1 file: mask and image combined)
  - <em>downloadButton2:</em> download the last segmented image as npy (2 files: one for the mask, one for the image)
  - <em>closeButton:</em> closes the Popup Window
  - <em>closeButton2:</em> close QuickGuide window
  
## Backend of the application. 
The app.py provides the following functions
Endpoints
- <em>/seg_image/box:</em> segments an image with a box as prompt
- <em>/seg_image/point:</em> segments an image with points as prompts
- <em>/download_npy:</em> downloads the image in the npy format
- <em>/download_mask_npy:</em> downloads the mask in the npy format
- <em>/get-api-key:</em> passes the api key from the Docker image environment variable to index.html

Helper function:
- <em>/draw_mask:</em> draws the mask on the image
- <em>/segmentImageBox and segmentImagePoint:</em> loads the segment Anything model in the correct version and processes that with image with respective prompt types
