import numpy as np

#from ._find_contours_cy import _get_contour_segments
from find_contours_cuda import _get_contour_segments

from launch_cuda_kernel import launch_kernel

from collections import deque

_param_options = ('high', 'low')


def find_contours_full(image, level=None,
                  fully_connected='low', positive_orientation='low',
                  *,
                  mask=None):
    
    if fully_connected not in _param_options:
        raise ValueError('Parameters "fully_connected" must be either '
                         '"high" or "low".')
    if positive_orientation not in _param_options:
        raise ValueError('Parameters "positive_orientation" must be either '
                         '"high" or "low".')
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("Input array must be at least 2x2.")
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported.')
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError('Parameters "array" and "mask"'
                             ' must have same shape.')
        if not np.can_cast(mask.dtype, bool, casting='safe'):
            raise TypeError('Parameter "mask" must be a binary array.')
        mask = mask.astype(np.uint8, copy=False)
    if level is None:
        level = (np.nanmin(image) + np.nanmax(image)) / 2.0


    #segments = _get_contour_segments(image.astype(np.float64), float(level),
    #                                 fully_connected == 'high', mask=mask)
    segments = _get_contour_segments(image.astype(np.float64), float(level))

    contours = _assemble_contours(segments)
    if positive_orientation == 'high':
        contours = [c[::-1] for c in contours]

    return contours


def find_contours_splitted( kernel, bufferSize, stream, args,
                            result_1x, result_1y, result_2x, result_2y, 
                            dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, 
                            NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, 
                            image, level=None,
                            fully_connected='low', positive_orientation='low', *, mask=None):
    
    if fully_connected not in _param_options:
        raise ValueError('Parameters "fully_connected" must be either '
                         '"high" or "low".')
    if positive_orientation not in _param_options:
        raise ValueError('Parameters "positive_orientation" must be either '
                         '"high" or "low".')
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("Input array must be at least 2x2.")
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported.')
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError('Parameters "array" and "mask"'
                             ' must have same shape.')
        if not np.can_cast(mask.dtype, bool, casting='safe'):
            raise TypeError('Parameter "mask" must be a binary array.')
        mask = mask.astype(np.uint8, copy=False)
    if level is None:
        level = (np.nanmin(image) + np.nanmax(image)) / 2.0


    #segments = _get_contour_segments(image.astype(np.float64), float(level),
    #                                 fully_connected == 'high', mask=mask)
    #segments = _get_contour_segments(image.astype(np.float64), float(level))
    segments = launch_kernel(   kernel, bufferSize, stream, args,
                                result_1x, result_1y, result_2x, result_2y, 
                                dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, 
                                NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, 
                                image.astype(np.float64), float(level))

    contours = _assemble_contours(segments)
    if positive_orientation == 'high':
        contours = [c[::-1] for c in contours]
    return contours


def _assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        # Ignore degenerate segments.
        # This happens when (and only when) one vertex of the square is
        # exactly the contour level, and the rest are above or below.
        # This degenerate vertex will be picked up later by neighboring
        # squares.
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None)) # Pop an element not present from the dictionary, provided a default value
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            # We need to connect these two contours.
            if tail is head:
                # We need to closed a contour: add the end point
                head.append(to_point)
            else:  # tail is not head
                # We need to join two distinct contours.
                # We want to keep the first contour segment created, so that
                # the final contours are ordered left->right, top->bottom.
                if tail_num > head_num:
                    # tail was created second. Append tail to head.
                    head.extend(tail)
                    # Remove tail from the detected contours
                    contours.pop(tail_num, None)
                    # Update starts and ends
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:  # tail_num <= head_num
                    # head was created second. Prepend head to tail.
                    tail.extendleft(reversed(head))
                    # Remove head from the detected contours
                    starts.pop(head[0], None)  # head[0] can be == to_point!
                    contours.pop(head_num, None)
                    # Update starts and ends
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            # We need to add a new contour
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:  # tail is not None
            # tail first element is to_point: the new segment should be
            # prepended.
            tail.appendleft(from_point)
            # Update starts
            starts[from_point] = (tail, tail_num)
        else:  # tail is None and head is not None:
            # head last element is from_point: the new segment should be
            # appended
            head.append(to_point)
            # Update ends
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]
