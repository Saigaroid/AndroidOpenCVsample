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
        val gray = Mat()
        val hsv = Mat()

        // グレースケール変換
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        // HSV変換
        Imgproc.cvtColor(rgba, hsv, Imgproc.COLOR_RGBA2HSV)

        // ガウシアンブラーでノイズを低減
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // コントラストを調整するためのCLAHE
        val clahe = Imgproc.createCLAHE()
        clahe.clipLimit = 2.0
        clahe.apply(gray, gray)

        // ヒストグラム均等化
        Imgproc.equalizeHist(gray, gray)

        // 適応的閾値処理
        val thresh = Mat()
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)

        // エッジ検出（Cannyアルゴリズムを使用）
        val edges = Mat()
        Imgproc.Canny(thresh, edges, 50.0, 150.0)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        // デバッグ用ログを追加
        Log.d(TAG, "Contours found: ${contours.size}")

        // 画像の面積を取得
        val imageArea = rgba.rows() * rgba.cols()

        for (contour in contours) {
            val approxCurve = MatOfPoint2f()
            val contour2f = MatOfPoint2f(*contour.toArray())
            val approxDistance = Imgproc.arcLength(contour2f, true) * 0.02
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true)

            // 輪郭の近似結果のログ
            Log.d(TAG, "ApproxPolyDP result points: ${approxCurve.total()}")

            // 輪郭の面積を計算
            val contourArea = Imgproc.contourArea(approxCurve)

            // 面積が画像全体の一部であることを確認 (例: 画像面積の10%以上50%以下)
            if (approxCurve.total() == 4L && contourArea > 1000 && contourArea < imageArea * 0.5) {
                val points = MatOfPoint(*approxCurve.toArray())
                if (isRectangle(points.toArray())) {
                    val aspectRatio = getAspectRatio(points.toArray())
                    if (isValidAspectRatio(aspectRatio)) {
                        Imgproc.drawContours(rgba, listOf(points), -1, Scalar(0.0, 255.0, 0.0), 3)
                    }
                }
            }
        }

        // メモリ解放
        gray.release()
        thresh.release()
        edges.release()
        hierarchy.release()
        hsv.release()

        return rgba
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

        // 角度のコサインが約90度 (cosineが0に近い)であることを確認
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
        // 一般的なIDカードのアスペクト比は約1.58（85.60mm / 53.98mm）
        val lowerBound = 1.4
        val upperBound = 1.8
        return aspectRatio > lowerBound && aspectRatio < upperBound
    }
}
