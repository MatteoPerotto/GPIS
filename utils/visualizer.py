import copy
import queue
from itertools import product
from multiprocessing import Queue, Process
from multiprocessing import Barrier
import torch
import vispy.visuals
from vispy import app, scene
from vispy.scene import ViewBox
from vispy.scene.visuals import Markers
import numpy as np
from vispy.visuals import LineVisual

import matplotlib as mpl
import matplotlib.cm as cm
vispy.use("osmesa")

class Visualizer(Process):

    def __init__(self):
        super().__init__()
        self.name = 'Visualizer'
        self.setup_f = False
        self.barrier = Barrier(2)

        self.queue_in = Queue(1)

        self.pred_f = True
        self.gt_f = True
        self.ref_f = True

        self.last_data = None

    def run(self):
        self.build_gui()
        self.barrier.wait()
        app.run()

    def build_gui(self):

        # logger.debug('Started gui building')
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 1200, 600
        canvas.show()

        self.grid = canvas.central_widget.add_grid()
        canvas.events.key_press.connect(self.dispatch)

        vb = ViewBox()
        vb = self.grid.add_widget(vb, row=0, col=0, row_span=1, col_span=1)

        vb.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1)
        vb.border_color = (0.5, 0.5, 0.5, 1)

        self.scatter1 = Markers(parent=vb.scene)
        self.scatter2 = Markers(parent=vb.scene)

        v = np.array(list((product([-0.5, 0.5], repeat=3))))
        e = np.arange(0, 8).reshape(-1, 2)
        e = np.concatenate([e, np.array([[0, 2], [0, 4], [2, 6], [4, 6]])])
        e = np.concatenate([e, np.array([[0, 2], [0, 4], [2, 6], [4, 6]]) + 1])
        self.cube = scene.Line(v, color='orange', connect=e, width=10, parent=vb.scene)

        self.ref = scene.XYZAxis(parent=vb.scene, width=10)

        # logger.debug('Gui built successfully')

    def dispatch(self, event):
        if event.text == 'p':
            self.pred_f = not self.pred_f
        elif event.text == 'g':
            self.gt_f = not self.gt_f
        elif event.text == 'r':
            self.ref_f = not self.ref_f
            if self.ref_f:
                self.cube.set_data(width=10)
                self.ref.set_data(width=10)
            else:
                self.cube.set_data(width=0)
                self.ref.set_data(width=0)
            return
        else:
            return

        if self.last_data is not None:
            try:
                self.queue_in.put(self.last_data, block=False)
            except queue.Full:
                return

    def on_timer(self, _):

        if not self.queue_in.empty():
            data = self.queue_in.get()
            self.last_data = data

            predictions, var, train = data['predictions'], data['variance'], data['train']

            norm = mpl.colors.Normalize(vmin=predictions[:, 0].min(), vmax=predictions[:, 0].max())
            m = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)
            colors = m.to_rgba(predictions[:, 0])[..., :-1]

            if self.pred_f:
                self.scatter1.set_data(predictions, edge_color=colors,
                                      face_color=colors, size=5)
            else:
                self.scatter1.set_data(np.array([[0, 0, 0]]))

            colors_complete = np.array([[0, 1, 0]]).repeat(train.shape[0], 0) #np.concatenate([np.array([[0, 1, 0]]).repeat(train.shape[0], 0),
                                       #np.array([[1, 0, 0]]).repeat(hidden.shape[0], 0)])
            complete = train # np.concatenate([train, hidden])

            if self.gt_f:
                self.scatter2.set_data(complete, edge_color=colors_complete,
                                      face_color=colors_complete, size=10)
            else:
                self.scatter2.set_data(np.array([[0, 0, 0]]))

    def update(self, data):
        if not self.setup_f:
            self.setup_f = True
            self.start()
            self.barrier.wait()

        if self.queue_in.empty():
            self.queue_in.put(copy.deepcopy(data))

    def stop(self):
        app.quit()

    def on_draw(self, event):
        pass
