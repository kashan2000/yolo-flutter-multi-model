// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import android.view.View
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import java.io.ByteArrayOutputStream
import java.util.ArrayList
import java.util.HashMap

/**
 * YoloPlatformView - Native view bridge from Flutter
 */
class YoloPlatformView(
    private val context: Context,
    private val viewId: Int,
    creationParams: Map<String?, Any?>?,
    private val streamHandler: EventChannel.StreamHandler,
    private val methodChannel: MethodChannel?,
    private val factory: YoloPlatformViewFactory
) : PlatformView, MethodChannel.MethodCallHandler {

    private val yoloView: YoloView = YoloView(context)
    private val TAG = "YoloPlatformView"
    
    // Initialization flag
    private var initialized = false
    
    // Frame counter for tracking inference results
    private var frameNumberCounter: Long = 0
    
    // Unique ID to send to Flutter
    private val viewUniqueId: String
    
    init {
        val dartViewIdParam = creationParams?.get("viewId")
        viewUniqueId = dartViewIdParam as? String ?: viewId.toString()

        try {
            // Parse model configurations from creation params
            val models = creationParams?.get("models") as? List<Map<String, Any>> ?: emptyList()
            
            // Initialize each model
            models.forEach { modelConfig ->
                val modelPath = modelConfig["modelPath"] as? String ?: "yolo11n"
                val taskString = modelConfig["task"] as? String ?: "detect"
                val confidenceThreshold = modelConfig["confidenceThreshold"] as? Double ?: 0.5
                val iouThreshold = modelConfig["iouThreshold"] as? Double ?: 0.45

                val task = YOLOTask.valueOf(taskString.uppercase())
                
                // Create predictor based on task
                val predictor = when (task) {
                    YOLOTask.DETECT -> ObjectDetector(context, modelPath, YoloFileUtils.loadLabelsFromAppendedZip(context, modelPath) ?: emptyList(), useGpu = true).apply {
                        setConfidenceThreshold(confidenceThreshold.toFloat())
                        setIouThreshold(iouThreshold.toFloat())
                    }
                    YOLOTask.SEGMENT -> Segmenter(context, modelPath, YoloFileUtils.loadLabelsFromAppendedZip(context, modelPath) ?: emptyList(), useGpu = true)
                    YOLOTask.POSE -> PoseEstimator(context, modelPath, YoloFileUtils.loadLabelsFromAppendedZip(context, modelPath) ?: emptyList(), useGpu = true)
                    else -> ObjectDetector(context, modelPath, YoloFileUtils.loadLabelsFromAppendedZip(context, modelPath) ?: emptyList(), useGpu = true)
                }

                // Add predictor to YoloView
                yoloView.addPredictor(predictor)
            }

            // Set up the method channel handler
            methodChannel?.setMethodCallHandler(this)

            // Attempt to initialize camera as soon as the view is created
            Log.d(TAG, "Attempting early camera initialization for YoloView.")
            yoloView.initCamera()

            // If context is already a LifecycleOwner, inform YoloView immediately
            if (context is LifecycleOwner) {
                Log.d(TAG, "Initial context is a LifecycleOwner, notifying YoloView.")
                yoloView.onLifecycleOwnerAvailable(context)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YoloPlatformView", e)
        }
    }
    
    // Handle method calls from Flutter
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            Log.d(TAG, "Received method call: ${call.method} with arguments: ${call.arguments}")
            
            when (call.method) {
                "setThreshold" -> {
                    val threshold = call.argument<Double>("threshold") ?: 0.5
                    Log.d(TAG, "Setting confidence threshold to $threshold")
                    yoloView.setConfidenceThreshold(threshold)
                    result.success(null)
                }
                "setConfidenceThreshold" -> {
                    val threshold = call.argument<Double>("threshold") ?: 0.5
                    Log.d(TAG, "Setting confidence threshold to $threshold")
                    yoloView.setConfidenceThreshold(threshold)
                    result.success(null)
                }
                // Support both "setIoUThreshold" (from Dart) and "setIouThreshold" (internal method)
                "setIoUThreshold", "setIouThreshold" -> {
                    val threshold = call.argument<Double>("threshold") ?: 0.45
                    Log.d(TAG, "Setting IoU threshold to $threshold")
                    yoloView.setIouThreshold(threshold)
                    result.success(null)
                }
                "setNumItemsThreshold" -> {
                    val numItems = call.argument<Int>("numItems") ?: 30
                    Log.d(TAG, "Setting numItems threshold to $numItems")
                    yoloView.setNumItemsThreshold(numItems)
                    result.success(null)
                }
                "setThresholds" -> {
                    val confidenceThreshold = call.argument<Double>("confidenceThreshold")
                    val iouThreshold = call.argument<Double>("iouThreshold")
                    val numItemsThreshold = call.argument<Int>("numItemsThreshold")
                    
                    if (confidenceThreshold != null) {
                        Log.d(TAG, "Setting confidence threshold to $confidenceThreshold")
                        yoloView.setConfidenceThreshold(confidenceThreshold)
                    }
                    if (iouThreshold != null) {
                        Log.d(TAG, "Setting IoU threshold to $iouThreshold")
                        yoloView.setIouThreshold(iouThreshold)
                    }
                    if (numItemsThreshold != null) {
                        Log.d(TAG, "Setting numItems threshold to $numItemsThreshold")
                        yoloView.setNumItemsThreshold(numItemsThreshold)
                    }
                    
                    result.success(null)
                }
                "switchCamera" -> {
                    Log.d(TAG, "Switching camera")
                    yoloView.switchCamera()
                    result.success(null)
                }
                "setShowUIControls" -> {
                    // Android doesn't have UI controls like iOS, so we just acknowledge the call
                    Log.d(TAG, "setShowUIControls called, but not applicable for Android")
                    result.success(null)
                }
                else -> {
                    Log.w(TAG, "Method not implemented: ${call.method}")
                    result.notImplemented()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling method call: ${call.method}", e)
            result.error("method_call_error", "Error handling method call: ${e.message}", null)
        }
    }
    
    // Convert YOLOResult to a Map for sending to Flutter
    private fun convertResultToMap(result: YOLOResult): Map<String, Any> {
        val map = HashMap<String, Any>()
        val detections = ArrayList<Map<String, Any>>()
        
        // Convert detection boxes
        for (box in result.boxes) {
            val detection = HashMap<String, Any>()
            detection["classIndex"] = box.index
            detection["className"] = box.cls
            detection["confidence"] = box.conf.toDouble()
            
            // Bounding box in original coordinates
            val boundingBox = HashMap<String, Any>()
            boundingBox["left"] = box.xywh.left.toDouble()
            boundingBox["top"] = box.xywh.top.toDouble()
            boundingBox["right"] = box.xywh.right.toDouble()
            boundingBox["bottom"] = box.xywh.bottom.toDouble()
            detection["boundingBox"] = boundingBox
            
            // Normalized bounding box (0-1)
            val normalizedBox = HashMap<String, Any>()
            normalizedBox["left"] = box.xywhn.left.toDouble()
            normalizedBox["top"] = box.xywhn.top.toDouble()
            normalizedBox["right"] = box.xywhn.right.toDouble()
            normalizedBox["bottom"] = box.xywhn.bottom.toDouble()
            detection["normalizedBox"] = normalizedBox
            
            detections.add(detection)
        }
        
        map["detections"] = detections
        
        // Convert speed metrics
        val processingTimeMs = result.speed.toDouble() // This should now be actual ms from ObjectDetector
        map["processingTimeMs"] = processingTimeMs
        
        // Use the (potentially smoothed) FPS from YOLOResult
        map["fps"] = result.fps?.toDouble() ?: 0.0
        
        // Optionally convert annotated image if available
        result.annotatedImage?.let { bitmap ->
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            map["annotatedImage"] = outputStream.toByteArray()
        }

        // Log the structure of the sent data for debugging
        Log.d(TAG, "Sending detection data to Flutter: ${detections.size} boxes")
        if (detections.isNotEmpty()) {
            val firstDetection = detections.first()
            Log.d(TAG, "First detection keys: ${firstDetection.keys.joinToString()}")
            Log.d(TAG, "First detection class: ${firstDetection["className"]}, confidence: ${firstDetection["confidence"]}")
        }
        
        return map
    }

    override fun getView(): View {
        Log.d(TAG, "Getting view: ${yoloView.javaClass.simpleName}")
        
        // Check if context is a LifecycleOwner
        if (context is androidx.lifecycle.LifecycleOwner) {
            val lifecycleOwner = context as androidx.lifecycle.LifecycleOwner
            Log.d(TAG, "Context is a LifecycleOwner with state: ${lifecycleOwner.lifecycle.currentState}")
        } else {
            Log.e(TAG, "Context is NOT a LifecycleOwner! This may cause camera issues.")
        }
        
        // Try setting custom layout parameters
        if (yoloView.layoutParams == null) {
            yoloView.layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT
            )
            Log.d(TAG, "Set layout params for YoloView")
        }
        
        return yoloView
    }

    override fun dispose() {
        Log.d(TAG, "Disposing YoloPlatformView for viewId: $viewId")
        // Clean up resources
        methodChannel?.setMethodCallHandler(null)
        // Notify the factory that this view is disposed
        factory.onPlatformViewDisposed(viewId)
    }

    /**
     * Called by YoloPlatformViewFactory when the Activity (which is a LifecycleOwner)
     * becomes available or changes.
     */
    fun notifyLifecycleOwnerAvailable(owner: LifecycleOwner) {
        Log.d(TAG, "LifecycleOwner (${owner.javaClass.simpleName}) is now available for viewId: $viewId. Notifying YoloView.")
        yoloView.onLifecycleOwnerAvailable(owner)
    }
        
        // Called by YoloPlugin to delegate permission results
        fun passRequestPermissionsResult(
            requestCode: Int,
            permissions: Array<String>, 
            grantResults: IntArray
        ) {
            Log.d(TAG, "passRequestPermissionsResult called in YoloPlatformView for viewId $viewId, delegating to yoloView")
            yoloView.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    
    /**
     * Resolves a model path that might be relative to app's internal storage
     * @param context Application context
     * @param modelPath The model path from Flutter
     * @return Resolved absolute path or original asset path
     */
    private fun resolveModelPath(context: Context, modelPath: String): String {
        // If it's already an absolute path, return it
        if (YoloUtils.isAbsolutePath(modelPath)) {
            return modelPath
        }
        
        // Check if it's a relative path to internal storage
        if (modelPath.startsWith("internal://")) {
            val relativePath = modelPath.substring("internal://".length)
            return "${context.filesDir.absolutePath}/$relativePath"
        }
        
        // Otherwise, consider it an asset path
        return modelPath
    }
}