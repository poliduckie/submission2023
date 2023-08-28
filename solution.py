import io

import cv2
import numpy as np
from PIL import Image

from aido_schemas import (
    Context,
    DB20Commands,
    DB20ObservationsWithTimestamp,
    DB20OdometryWithTimestamp,
    EpisodeStart,
    GetCommands,
    JPGImageWithTimestamp,
    LEDSCommands,
    logger,
    PWMCommands,
    RGB,
)

from poliduckie_segmentation import BirdEyeView
from poliduckie_segmentation.segmentation import Segmentation
from poliduckie_segmentation.line_extraction import LineExtraction

LineExtraction.N_CLUSTERS = 10
LineExtraction.MIN_ENTRIES = 8
LineExtraction.MAX_DISTANCE = 40
VERTICAL_CUTOFF = 150
STEERING_CLIPPING = 3

# TODO FIX
matrixPath = '/code/BirdEyeMatrices/SegmentationOutput.pkl'

seg = Segmentation()
birdeye = BirdEyeView(path = matrixPath)
lineExtraction_ = LineExtraction()

def segmentation(image):
  """
    Returns the image from the simulation with the segmentation

    Image: np.Array
    ------
    Image: np.Array
  """
  resized = cv2.resize(image, (320, 240))
  return seg.predict(resized)


def birdEyeTransform(image):
  """
    Returns the image from the simulation with the bird eye view

    Image: np.Array
    ------
    Image: np.Array
  """
  return birdeye.computeBirdEye(image)


def lineExtraction(image, n_points=100, mode="bezier", bezier_degree=3, img_height=240, vertical_cutoff=150):
  """
    Returns the central line extracted as a numpy array. The mode can be either "bezier" or "spline"

    Image: np.Array
    n_points: int
    mode: str
    bezier_degree: int
    ------
    Line: np.Array
  """
  if(np.nonzero(image)[0].size < LineExtraction.N_CLUSTERS):
    return np.array([[320/2], [img_height-vertical_cutoff]])
  
  if mode == "bezier":
    bezier_points = lineExtraction_.bezier_fit(image, degree=bezier_degree, nPoints=n_points, usePCA=True)
    
    if bezier_points[1, -1] > bezier_points[1, 0]:
      bezier_points = bezier_points[:, ::-1]

    return bezier_points
  
  elif mode == "spline":
    spline_points = lineExtraction_.spline_interpolation(image, nPoints=n_points)
    return spline_points

METER_PER_PIXEL = 0.4 / 150
CAR_POSITION = [320/2, 0]

def computeReference(line, N, vertical_cutoff, img_size_y):
  """
    Returns the references in meters
    Reference is a list with dimension (2, N)
    N is the prediction horizon of the MPC

    Line: np.Array
    N: int
    ------
    reference: List[[List[Float], List[Float]]]
  """
  reference = line[:, :N].copy()

  #center around car position
  reference[0] = [x - CAR_POSITION[0] for x in reference[0]]
  reference[1] = [img_size_y - vertical_cutoff - y  for y in reference[1]]

  #convert to meters
  reference[0] = [x * METER_PER_PIXEL for x in reference[0]]
  reference[1] = [y * METER_PER_PIXEL for y in reference[1]]
  
  # reference = [[], []]
  # reference[0] = [0 for x in range(N)]
  # reference[1] = [y for y in range(N)]
  return np.array(reference)

from poliduckie_segmentation.control import MPC
N = 10
M = MPC(N=10)

def runMpc(state, reference):
  """
    Returns the control inputs to give to the car.
    State is [x, y, theta, v, w]
    Reference is a list with dimension (2, N)

    state: [Float, Float, Float, Float, Float]
    reference: List[[List[Float], List[Float]]]
    ------
    u: List[Float, Float]
  """
  return M.mpc(state, reference)

from poliduckie_segmentation.model import Model
import casadi as ca

class RandomAgent:
    n: int

    def init(self, context: Context):
        self.n = 0
        self.state = ca.DM([0,0,np.pi / 2,0,0])
        self.F = Model()
        self.reference = []
        context.info("init()")

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}"')
        logger.info(data=data)

    def on_received_observations(self, context: Context, data: DB20ObservationsWithTimestamp):
        profiler = context.get_profiler()
        camera: JPGImageWithTimestamp = data.camera
        odometry: DB20OdometryWithTimestamp = data.odometry
        context.info(f"camera timestamp: {camera.timestamp}")
        context.info(f"odometry timestamp: {odometry.timestamp}")
        with profiler.prof("jpg2rgb"):
            _rgb = jpg2rgb(camera.jpg_data)
        image_BGR = cv2.cvtColor(_rgb, cv2.COLOR_RGB2BGR)
        #Segmentation
        image_segmentation = (segmentation(image_BGR)[0]*255).clip(0, 255).astype(np.uint8)
        img_size_y = image_segmentation.shape[0]

        #Bird eye view
        image_birdeye = birdEyeTransform(image_segmentation).clip(0, 255).astype(np.uint8)

        #Line extraction
        dottedLineMask = cv2.threshold(image_birdeye[:, :, 1], 70, 255, cv2.THRESH_BINARY)[1]
        #imshow(dottedLineMask, ax = axs[1,0], show=False, no_axis=False)
        dottedLineMaskCropped = dottedLineMask[VERTICAL_CUTOFF:, :]
        self.line = lineExtraction(dottedLineMaskCropped, mode='bezier', n_points=N+1, img_height=img_size_y, vertical_cutoff=VERTICAL_CUTOFF)

        #Compute reference
        self.reference = computeReference(self.line, N+1, VERTICAL_CUTOFF, img_size_y)


    def on_received_get_commands(self, context: Context, data: GetCommands):
        self.n += 1

        if self.reference == []:
            return
        
        #Compute action
        action = runMpc(state, self.reference)

        #Clip the steering
        action[1] = np.clip(action[1], -STEERING_CLIPPING, STEERING_CLIPPING)

        pwm_left, pwm_right = action

        #Update state
        state = self.F.step(*state.toarray().reshape(-1), action)

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write("commands", commands)

    def finish(self, context: Context):
        context.info("finish()")


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """Reads JPG bytes as RGB"""

    im = Image.open(io.BytesIO(image_data))
    im = im.convert("RGB")
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data
