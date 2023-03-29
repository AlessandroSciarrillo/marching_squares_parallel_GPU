class SegmentationStage(Stage):
    @staticmethod
    def contour_to_polygon(c):
        return Polygon(c)

    @staticmethod
    def contour_to_enclosing_circle(c):
        (c_x, c_y), radius = cv2.minEnclosingCircle(c)
        return Circle(c_x, c_y, 2 * radius)

    @staticmethod
    def contour_to_inner_circle(c):
        (c_x, c_y), size, _ = cv2.minAreaRect(c)
        return Circle(c_x, c_y, np.min(size))

    @staticmethod
    def contour_to_rect(c):
        x, y, w, h = cv2.boundingRect(c)
        return Rect(x + .5 * w, y + .5 * h, w, h)

    @staticmethod
    def contour_to_ellipse(c):
        (c_x, c_y), (s_w, s_h), a_deg = cv2.fitEllipse(c)
        return Ellipse(c_x, c_y, s_w, s_h, a_deg * np.pi / 180.)

    @staticmethod
    def contour_to_rotated_rect(c):
        (c_x, c_y), (s_w, s_h), a_deg = cv2.minAreaRect(c)
        return RotatedRect(c_x, c_y, s_w, s_h, a_deg * np.pi / 180.)

    @property
    def skip_channels(self):
        return sorted(self._skip_channels)

    @skip_channels.setter
    def skip_channels(self, chs):
        if chs is None or len(chs) == 0:
            self._skip_channels = set()
        elif type(chs) in { list, set }:
            self._skip_channels = set(chs)
        else:
            self._skip_channels = { int(x) for x in chs.replace(',', ' ').split(' ') if len(x) > 0 }

    CONTOUR_TO_SHAPE = {
        aliquis_pb2.SegmentationParameter.POLYGON: contour_to_polygon,
        aliquis_pb2.SegmentationParameter.INNER_CIRCLE: contour_to_inner_circle,
        aliquis_pb2.SegmentationParameter.OUTER_CIRCLE: contour_to_enclosing_circle,
        aliquis_pb2.SegmentationParameter.RECTANGLE: contour_to_rect,
        aliquis_pb2.SegmentationParameter.ELLIPSE: contour_to_ellipse,
        aliquis_pb2.SegmentationParameter.ROTATED_RECTANGLE: contour_to_rotated_rect
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported = [ 'type', 'shape', 'threshold', 'num', 'convex_hull', 'approx_epsilon', 'spline_degree', 'spline_smoothness', 'skip_channels' ]
        self.skip_channels = self.__getattr__('skip_channels')

        self.FIND_SHAPES = {
            aliquis_pb2.SegmentationParameter.CONTOURS: self.find_shapes_contours,
            aliquis_pb2.SegmentationParameter.MARCHING_SQUARES: self.find_shapes_marching_squares,
            aliquis_pb2.SegmentationParameter.DISTANCE_TRANSFORM: self.find_shapes_distance_transform
        }

    def fit_shapes(self, contours):
        def tospline(c):
            # TODO: fix
            points = Polygon(c.squeeze()).spline(self.spline_degree, self.spline_smoothness, self.spline_step).points
            return points.reshape(len(points), 1, 2)

        if self.spline_degree != 0 or self.spline_smoothness != 0:
            contours = map(tospline, contours)

        if self.approx_epsilon != 0:
            contours = [ cv2.approxPolyDP(c, self.approx_epsilon, True) for c in contours ]

        if self.convex_hull:
            contours = map(cv2.convexHull, contours)

        return list(map(SegmentationStage.CONTOUR_TO_SHAPE[self.shape].__func__, contours))

    def fit_shape_distance_transform(self, ccomp_mask, p_x, p_y, length):
        w = 2 * np.linalg.norm(length)
        if self.shape == aliquis_pb2.SegmentationParameter.INNER_CIRCLE:
            cv2.circle(ccomp_mask, (int(round(p_x)), int(round(p_y))), int(round(.5 * w)), 0, -1)
            return Circle(p_x, p_y, w)
        else:
            theta = np.arctan2(length[1], length[0])
            h = np.abs(2 * get_mask_minradius(ccomp_mask, (p_x, p_y), .5 * np.pi + theta))
            cv2.ellipse(ccomp_mask, ((int(round(p_x)), int(round(p_y))), (int(round(w)), int(round(h))), theta * 180. / np.pi), 0, -1)

            if self.shape == aliquis_pb2.SegmentationParameter.OUTER_CIRCLE:
                return Circle(p_x, p_y, np.max([ w, h ]))
            elif self.shape == aliquis_pb2.SegmentationParameter.ELLIPSE:
                return Ellipse(p_x, p_y, w, h, theta)
            elif self.shape == aliquis_pb2.SegmentationParameter.ROTATED_RECTANGLE:
                return RotatedRect(p_x, p_y, w, h, theta)
            elif self.shape == aliquis_pb2.SegmentationParameter.RECTANGLE:
                return RotatedRect(p_x, p_y, w, h, theta).to_rect()
            else: #if self.shape == aliquis_pb2.SegmentationParameter.POLYGON:
                return Polygon(RotatedRect(p_x, p_y, w, h, theta).quad)

    def filter_shapes(self, shapes):
        if self.filter == aliquis_pb2.SegmentationParameter.ANY or len(shapes) == 0:
            return shapes

        # Adjust thresholds
        area_threshold_min = max(0, self.area_threshold_min)
        area_threshold_max = float('inf') if self.area_threshold_max <= 0 else self.area_threshold_max

        # Pre-compute areas
        areas = [ shape.area for shape in shapes ]

        # Filter areas
        if self.filter == aliquis_pb2.SegmentationParameter.AREA_MAX:
            return [ shapes[np.argmax(areas)] ]
        elif self.filter == aliquis_pb2.SegmentationParameter.AREA_MIN:
            return [ shapes[np.argmin(areas)] ]
        else: # if self.filter == aliquis_pb2.SegmentationParameter.AREA_THRESHOLD:
            return [ c for c, a in zip(shapes, areas) if area_threshold_min <= a <= area_threshold_max ]

    def find_shapes_contours(self, im):
        #loop = asyncio.get_running_loop()
        #with futures.ThreadPoolExecutor(1) as pool:
        #    contours, _ = await loop.run_in_executor(pool, cv2.findContours, np.uint8(im > self.threshold) if self.threshold > 0 else im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(np.uint8(im > self.threshold) if self.threshold > 0 else im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours, _ = await loop.run_in_executor(None, cv2.findContours, np.uint8(im > self.threshold) if self.threshold > 0 else im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self.filter_shapes(self.fit_shapes([ c.squeeze(1).astype(np.float32) for c in contours ]))

    def find_shapes_marching_squares(self, im):
        contours = []
        if im.ndim == 3:
            im = im.squeeze(2)
        for cnt_yx in find_contours(np.pad(im, ((1, 1), (1, 1))), self.threshold):
        #for cnt_yx in find_contours(im, self.threshold):
            cnt = cnt_yx[:, ::-1].astype(np.float32) - 1
            #cnt = cnt_yx[:, ::-1].astype(np.float32)

            # We only search for contours of islands of high-values, so create the patch if and
            # only if oriented area is positive, i.e. contour orientation is clockwise
            if cv2.contourArea(cnt, oriented=True) > 0:
                contours.append(cnt)
        return self.filter_shapes(self.fit_shapes(contours))

    def find_shapes_distance_transform(self, im):
        mask = im.copy()
        dists_unmasked, labels = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        _, markers = cv2.connectedComponents(mask)
        dists = np.ma.masked_array(dists_unmasked, mask=~mask)
        labels_outer = labels * dists.mask

        # Adjust thresholds
        size_threshold_min = max(0, self.size_threshold_min)
        size_threshold_max = float('inf') if self.size_threshold_max <= 0 else self.size_threshold_max

        ccomp_segments = {}
        while True:
            max_y, max_x = np.unravel_index(np.argmax(dists), dists.shape)
            max_dist = dists[max_y, max_x]
            if not size_threshold_min <= max_dist < size_threshold_max:
                break

            max_cid = markers[max_y, max_x]
            max_p = np.float32([ max_x, max_y ])
            max_length = np.argwhere(labels_outer == labels[max_y, max_x])[0][::-1].astype(np.float32) - max_p
            seg = (max_p, max_length)

            try:
                ccomp_segments[max_cid].append(seg)
            except KeyError:
                ccomp_segments[max_cid] = [ seg ]

            cv2.circle(dists.mask.view(np.uint8), (max_x, max_y), max_dist, 1, -1)

        shapes = []
        for cid, segments in ccomp_segments.items():
            ccomp_mask = (markers == cid).view(np.uint8)
            if len(segments) == 1:
                shapes.append(self.fit_shape_distance_transform(ccomp_mask, segments[0][0][0], segments[0][0][1], segments[0][1]))
            elif self.separate:
                moments = cv2.moments(ccomp_mask, True)
                area = moments['m00']
                center = np.array([ moments['m10'] / area, moments['m01'] / area ])

                #
                segments.sort(key=lambda seg_ps: np.linalg.norm(seg_ps[0] - center), reverse=True)

                for (p_x, p_y), length in segments:
                    shapes.append(self.fit_shape_distance_transform(ccomp_mask, p_x, p_y, length))

        return self.filter_shapes(shapes)

    def feed(self, patches):
        find_shapes = self.FIND_SHAPES[self.type]
        if not self.on_values:
            # if len(patches) > 0 and self.threads > 0:
            #     loop = asyncio.get_running_loop()
            #     with futures.ThreadPoolExecutor(self.threads) as pool:
            #         done, _ = await asyncio.wait([ loop.run_in_executor(pool, find_shapes, p.im) for p in patches ])
            #         shapes = [ await coro for coro in done ]
            # else:
            shapes = [ find_shapes(p.im) for p in patches ]
            return [ p.addSubpatchWithShape(shape) for p, shapes in zip(patches, shapes) for shape in shapes ]

        ops = []
        #loop = asyncio.get_running_loop()
        for patch in patches:
            im = np.atleast_3d(patch.values)
            h, w, channels = im.shape

            # if self.threads > 0:
            #     loop = asyncio.get_running_loop()
            #     with futures.ThreadPoolExecutor(self.threads) as pool:
            #         done, _ = await asyncio.wait([ loop.run_in_executor(pool, find_shapes, im[:, :, c]) for c in range(channels) ])
            #         channels_shapes = [ await coro for coro in done ]
            # else:
            channels_shapes = { c: find_shapes(im[:, :, c]) for c in (set(range(channels)) - self._skip_channels) }

            for c, shapes in channels_shapes.items():
                for shape in shapes:
                    r = shape.to_rect()
                    if r.w <= 0 or r.h <= 0:
                        continue

                    # Extract patch from bounding box
                    x = max(0, int(r.x - .5 * r.w))
                    #w = min(im.shape[1], int(r.x + .5 * r.w + 0.5) - x + 1)
                    w = min(im.shape[1], int(r.x + .5 * r.w) + 1) - x
                    y = max(0, int(r.y - .5 * r.h))
                    #h = min(im.shape[0], int(r.y + .5 * r.h + 0.5) - y + 1)
                    h = min(im.shape[0], int(r.y + .5 * r.h) + 1) - y
                    nwp = patch.addSubpatchFromRect((y, x), (h, w))
                    nwp.shape = shape.apply(nwp.tr)

                    # Prepare a mask to draw polygon inside (inside = valid = 1, outside = invalid = 0)
                    mask = np.zeros(nwp.im.shape[:2], dtype=np.uint8)
                    nwp.shape.draw(mask, 0xff, -1)

                    # Set patch values as the pixel with max value on the current class inside the shape
                    class_hm = im[y:(y + nwp.im.shape[0]), x:(x + nwp.im.shape[1]), c] # load the heatmap of current class
                    mask[class_hm < self.threshold] = 0 # remove pixels under threshold
                    values_inside = class_hm[mask != 0] # select all pixel values of current class inside the shape

                    if self.value_type == aliquis_pb2.ContoursFinderParameter.RANKING:
                        # Sort and mean over the n maximum values
                        # As we only need to know the n maximum values and not their order
                        # np.partition should be faster than np.sort (O(n) vs O(n^2) or O(n*log(n)))
                        ranking = np.partition(values_inside, -self.num) if self.num < len(values_inside) else values_inside
                        p_value = np.sum(ranking[:(-self.num - 1):-1]) / self.num
                    else:
                        # find max value among pixels inside the shape
                        max_val = np.max(values_inside)

                        p_value = -np.inf
                        # iterate over each peak
                        for y_max, x_max in zip(*np.where(class_hm == max_val)):
                            if self.num == 0:
                                value = class_hm[y_max, x_max]
                            else:
                                if self.num == 4:
                                    yy = np.array([0, -1, 0, 1, 0]) + y_max
                                    xx = np.array([0, 0, 1, 0, -1]) + x_max
                                else: # self.neighbors == 8
                                    yy = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0]) + y_max
                                    xx = np.array([0, -1, 0, 1, 1, 1, 0, -1, -1]) + x_max
                                inside_mask = (0 <= yy) & (yy < height) & (0 <= xx) & (xx < width)
                                yy = yy[inside_mask] # only xx and yy
                                xx = xx[inside_mask] # inside the image

                                value = np.sum(class_hm[yy,xx]) / (self.num + 1)

                            # use max among all peaks
                            p_value = max(p_value, value)

                    nwp.values = np.zeros(channels, dtype=np.float32)
                    nwp.values[c] = p_value
                    ops.append(nwp)
        return ops