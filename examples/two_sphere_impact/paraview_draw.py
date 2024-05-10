# trace generated using paraview version 5.12.0-RC2
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 12


import os

curr_file_dir_path = os.path.dirname(os.path.realpath(__file__))


output_path = os.path.join(curr_file_dir_path, "./output/")

files_in_output = [os.path.join(output_path, file) for file in os.listdir(output_path)]

files_created_date = [os.path.getctime(file) for file in files_in_output]

sorted_files_in_output = [x for _, x in sorted(zip(files_created_date, files_in_output))]

particle_files = list(filter(lambda file: "particles" in file, sorted_files_in_output))


#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML PolyData Reader'
particles0vtp = XMLPolyDataReader(
    registrationName="particles0.vtp*",
    FileName=particle_files,
    # FileName=['/home/retief/Projects/PyroclastExamples/ball_drop/output/particles0.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles100.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles200.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles300.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles400.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles500.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles600.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles700.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles800.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles900.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1000.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1100.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1200.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1300.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1400.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1500.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1600.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1700.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1800.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles1900.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2000.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2100.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2200.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2300.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2400.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2500.vtp', '/home/retief/Projects/PyroclastExamples/ball_drop/output/particles2600.vtp']
)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on particles0vtp
particles0vtp.TimeArray = "None"

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# # show data in view
particles0vtpDisplay = Show(particles0vtp, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
particles0vtpDisplay.Representation = "Point Gaussian"

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# changing interaction mode based on data extents
renderView1.CameraPosition = [0.5, 0.5, 2.303125]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# change representation type
# particles0vtpDisplay.SetRepresentationType('Point Gaussian')

renderView1.ResetActiveCameraToNegativeZ()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# set scalar coloring
ColorBy(particles0vtpDisplay, ("POINTS", "Velocity", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
particles0vtpDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
particles0vtpDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction("Velocity")

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction("Velocity")

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D("Velocity")

# Rescale transfer function
velocityLUT.RescaleTransferFunction(0.0, 0.52)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(0.0, 0.52)

# Rescale 2D transfer function
velocityTF2D.RescaleTransferFunction(0.0, 0.52, 0.0, 1.0)

# Properties modified on particles0vtpDisplay
particles0vtpDisplay.GaussianRadius = 0.01

# Properties modified on particles0vtpDisplay
particles0vtpDisplay.ShaderPreset = "Plain circle"

# create a new 'XML PolyData Reader'
nodes0vtp = XMLPolyDataReader(registrationName="nodes0.vtp", FileName=[output_path + "/nodes0.vtp"])

# set active source
SetActiveSource(nodes0vtp)

# show data in view
nodes0vtpDisplay = Show(nodes0vtp, renderView1, "GeometryRepresentation")


# update the view to ensure updated data information
renderView1.Update()

# change representation type
nodes0vtpDisplay.SetRepresentationType("Outline")


# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1542, 779)

# current camera placement for renderView1
renderView1.InteractionMode = "2D"
renderView1.CameraPosition = [0.7, 0.5, 1.8782849302036033]
renderView1.CameraFocalPoint = [0.7, 0.5, 0.0]
renderView1.CameraParallelScale = 0.7117515888554667

# save animation
SaveAnimation(
    filename=curr_file_dir_path + "/postprocess/video/output_vid.avi",
    viewOrLayout=renderView1,
    location=16,
    ImageResolution=[1540, 776],
    FontScaling="Scale fonts proportionally",
    OverrideColorPalette="",
    StereoMode="No change",
    TransparentBackground=0,
    FrameRate=16,
    FrameStride=1,
    FrameWindow=[0, 138],
    # FFMPEG options
    Compression=5,
    Quality="1",
)


SaveAnimation(
    filename=curr_file_dir_path + "/postprocess/frames/frames.png",
    viewOrLayout=renderView1,
    location=16,
    ImageResolution=[1542, 779],
    FrameWindow=[0, 138],
    # PNG options
    CompressionLevel="8",
)
# #================================================================
# # addendum: following script captures some of the application
# # state to faithfully reproduce the visualization during playback
# #================================================================

# #--------------------------------
# # saving layout sizes for layouts

# # layout/tab size in pixels
# layout1.SetSize(1542, 779)

# #-----------------------------------
# # saving camera placements for views

# # current camera placement for renderView1
# renderView1.InteractionMode = '2D'
# renderView1.CameraPosition = [0.5, 0.5, 1.8782849302036033]
# renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
# renderView1.CameraParallelScale = 0.7117515888554667


# ##--------------------------------------------
# ## You may need to add some code at the end of this python script depending on your usage, eg:
# #
# ## Render all views to see them appears
# # RenderAllViews()
# #
# ## Interact with the view, usefull when running from pvpython
# # Interact()
# #
# ## Save a screenshot of the active view
# # SaveScreenshot("path/to/screenshot.png")
# #
# ## Save a screenshot of a layout (multiple splitted view)
# # SaveScreenshot("path/to/screenshot.png", GetLayout())
# #
# ## Save all "Extractors" from the pipeline browser
# # SaveExtracts()
# #
# ## Save a animation of the current active view
# # SaveAnimation()
# #
# ## Please refer to the documentation of paraview.simple
# ## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
# ##--------------------------------------------
