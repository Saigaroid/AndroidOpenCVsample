import org.opencv.android.CameraBridgeViewBase
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import android.util.Log

class MyOpenCVActivity : CameraBridgeViewBase.CvCameraViewListener2 {

    companion object {
        private const val TAG = "MyOpenCVActivity"
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val rgba = inputFrame.rgba()
        val processedFrame = preprocessFrame(rgba)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(processedFrame, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        Log.d(TAG, "Contours found: ${contours.size}")

        val imageArea = rgba.rows() * rgba.cols()

        for (contour in contours) {
            val approxCurve = MatOfPoint2f()
            val contour2f = MatOfPoint2f(*contour.toArray())
            val approxDistance = Imgproc.arcLength(contour2f, true) * 0.02
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true)

            Log.d(TAG, "ApproxPolyDP result points: ${approxCurve.total()}")

            val contourArea = Imgproc.contourArea(approxCurve)

            if (approxCurve.total() == 4L && contourArea > 1000 && contourArea < imageArea * 0.5) {
                val points = MatOfPoint(*approxCurve.toArray())
                if (isRectangle(points.toArray())) {
                    val aspectRatio = getAspectRatio(points.toArray())
                    if (isValidAspectRatio(aspectRatio)) {
                        Imgproc.drawContours(rgba, listOf(points), -1, Scalar(0.0, 255.0, 0.0), 3)
                    }
                }
            }
            approxCurve.release()
            contour2f.release()
        }

        processedFrame.release()
        hierarchy.release()

        return rgba
    }

    private fun preprocessFrame(rgba: Mat): Mat {
        val gray = Mat()
        val thresh = Mat()
        val edges = Mat()

        try {
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

            val clahe = Imgproc.createCLAHE()
            clahe.clipLimit = 2.0
            clahe.apply(gray, gray)

            Imgproc.equalizeHist(gray, gray)
            Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)
            Imgproc.Canny(thresh, edges, 50.0, 150.0)

        } finally {
            gray.release()
            thresh.release()
        }

        return edges
    }

    private fun isRectangle(points: Array<Point>): Boolean {
        if (points.size != 4) {
            return false
        }

        var maxCosine = 0.0
        for (i in points.indices) {
            val cosine = Math.abs(angle(points[i], points[(i + 1) % 4], points[(i + 2) % 4]))
            maxCosine = Math.max(maxCosine, cosine)
        }

        return maxCosine < 0.3
    }

    private fun angle(p1: Point, p2: Point, p3: Point): Double {
        val dx1 = p1.x - p2.x
        val dy1 = p1.y - p2.y
        val dx2 = p3.x - p2.x
        val dy2 = p3.y - p2.y
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))
    }

    private fun getAspectRatio(points: Array<Point>): Double {
        val width = points[1].x - points[0].x
        val height = points[2].y - points[1].y
        return width / height
    }

    private fun isValidAspectRatio(aspectRatio: Double): Boolean {
        val lowerBound = 1.4
        val upperBound = 1.8
        return aspectRatio > lowerBound && aspectRatio < upperBound
    }
}
