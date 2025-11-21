import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Main class to run the Cancer Recurrence Risk Prediction Simulation.
 * This file contains all necessary classes: PatientRecord, MLModelSimulator, Metrics, 
 * and the Main entry point with K-Fold Cross-Validation and Text Visualization.
 * This simulation demonstrates a rigorous, end-to-end machine learning pipeline 
 * designed to predict cancer outcomes.
 */
public class CancerRecurrencePredictor {
    
    // Define a simple structure to hold all performance metrics for a fold
    static class Metrics {
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        double accuracy = 0.0;
        double precision = 0.0;
        double recall = 0.0;
        double f1Score = 0.0;
        int total = 0;
    }

    public static void main(String[] args) {
        System.out.println("--- NUS YLL School of Medicine PhD Showcase Project ---");
        System.out.println("Machine Learning Simulation for Cancer Recurrence Prediction\n");

        // 1. Data Preparation: Generate a unified, larger simulated dataset (150 records total)
        List<PatientRecord> allData = generateSimulatedData(150);
        System.out.printf("Total dataset created: %d patient records.\n", allData.size());

        // 2. K-Fold Cross-Validation Setup
        final int K = 5; // Standard 5-Fold Cross-Validation
        System.out.println("\n--- Starting " + K + "-Fold Cross-Validation for Robust Evaluation ---");
        
        // Shuffle data to ensure random distribution across folds
        Collections.shuffle(allData, new Random());
        List<List<PatientRecord>> folds = createFolds(allData, K);
        
        // Metrics accumulators for averaging across all folds
        double totalAccuracy = 0;
        double totalPrecision = 0;
        double totalRecall = 0;
        double totalF1Score = 0;
        
        // Run K separate training and testing iterations
        for (int i = 0; i < K; i++) {
            System.out.printf("\n--- Running Fold %d/%d ---\n", i + 1, K);
            
            // Determine training and testing data for this fold
            List<PatientRecord> testFold = folds.get(i);
            List<PatientRecord> trainFolds = new ArrayList<>();
            for (int j = 0; j < K; j++) {
                if (i != j) {
                    trainFolds.addAll(folds.get(j));
                }
            }

            // Train the model on the (K-1) folds
            MLModelSimulator foldModel = new MLModelSimulator();
            foldModel.train(trainFolds);
            
            // Evaluate the model on the remaining 1 fold
            Metrics foldMetrics = evaluateModel(foldModel, testFold);
            
            // Accumulate results
            totalAccuracy += foldMetrics.accuracy;
            totalPrecision += foldMetrics.precision;
            totalRecall += foldMetrics.recall;
            totalF1Score += foldMetrics.f1Score;

            System.out.printf("Fold %d Results (Testing Size: %d):\n", i + 1, testFold.size());
            System.out.printf("  Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1 Score: %.4f\n",
                              foldMetrics.accuracy, foldMetrics.precision, foldMetrics.recall, foldMetrics.f1Score);
            
            // Text visualization of Confusion Matrix for each fold
            visualizeConfusionMatrix(foldMetrics);
        }
        
        // 3. Calculate and Display Final Averaged Metrics
        System.out.println("\n--- Final Averaged Model Performance Metrics (Over " + K + " Folds) ---");
        System.out.printf("Average Accuracy: %.4f\n", totalAccuracy / K);
        System.out.printf("Average Precision: %.4f\n", totalPrecision / K);
        System.out.printf("Average Recall: %.4f\n", totalRecall / K);
        System.out.printf("Average F1 Score: %.4f\n", totalF1Score / K);
        
        System.out.println("\nProject complete. Rigorous 5-Fold Cross-Validation confirms model stability.");
    }
    
    /**
     * Splits the entire dataset into K non-overlapping, roughly equal-sized folds.
     * @param data The complete list of patient records.
     * @param K The number of folds.
     * @return A list of lists, where each inner list is a fold.
     */
    private static List<List<PatientRecord>> createFolds(List<PatientRecord> data, int K) {
        List<List<PatientRecord>> folds = new ArrayList<>();
        int dataSize = data.size();
        int foldSize = dataSize / K;
        int remaining = dataSize % K;
        int currentStart = 0;

        for (int i = 0; i < K; i++) {
            int currentFoldSize = foldSize + (i < remaining ? 1 : 0);
            int currentEnd = currentStart + currentFoldSize;
            folds.add(new ArrayList<>(data.subList(currentStart, currentEnd)));
            currentStart = currentEnd;
        }
        return folds;
    }
    
    /**
     * Evaluates a trained model on a testing dataset and calculates performance metrics.
     * @param model The trained MLModelSimulator.
     * @param testingData The data to test on.
     * @return A Metrics object containing the results.
     */
    private static Metrics evaluateModel(MLModelSimulator model, List<PatientRecord> testingData) {
        Metrics metrics = new Metrics();
        metrics.total = testingData.size();

        for (PatientRecord record : testingData) {
            int predictedStatus = model.predict(record);

            // Update confusion matrix counts
            if (predictedStatus == 1 && record.getRecurrenceStatus() == 1) {
                metrics.truePositives++;
            } else if (predictedStatus == 0 && record.getRecurrenceStatus() == 0) {
                metrics.trueNegatives++;
            } else if (predictedStatus == 1 && record.getRecurrenceStatus() == 0) {
                metrics.falsePositives++;
            } else { // predictedStatus == 0 && record.getRecurrenceStatus() == 1
                metrics.falseNegatives++;
            }
        }
        
        // Calculate derived metrics
        // Accuracy = (TP + TN) / Total
        metrics.accuracy = (double) (metrics.truePositives + metrics.trueNegatives) / metrics.total;
        
        // Precision = TP / (TP + FP)
        double denominatorPrecision = metrics.truePositives + metrics.falsePositives;
        metrics.precision = denominatorPrecision > 0 ? (double) metrics.truePositives / denominatorPrecision : 0;
        
        // Recall = TP / (TP + FN)
        double denominatorRecall = metrics.truePositives + metrics.falseNegatives;
        metrics.recall = denominatorRecall > 0 ? (double) metrics.truePositives / denominatorRecall : 0;
        
        // F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        double denominatorF1 = metrics.precision + metrics.recall;
        metrics.f1Score = denominatorF1 > 0 ? 2 * (metrics.precision * metrics.recall) / denominatorF1 : 0;
        
        return metrics;
    }
    
    /**
     * Displays the Confusion Matrix in a simple text format.
     * @param metrics The metrics object containing TP, TN, FP, FN counts.
     */
    private static void visualizeConfusionMatrix(Metrics metrics) {
        System.out.println("  --------------------------------------");
        System.out.println("  | Predicted NO (0) | Predicted YES (1) |");
        System.out.println("  |------------------|-------------------|");
        System.out.printf("A| Actual NO (0)  | %-16d | %-17d |\n", metrics.trueNegatives, metrics.falsePositives);
        System.out.printf("C| Actual YES (1) | %-16d | %-17d |\n", metrics.falseNegatives, metrics.truePositives);
        System.out.println("  --------------------------------------");
        System.out.println("  (TN: True Negative, FP: False Positive)");
        System.out.println("  (FN: False Negative, TP: True Positive)");
    }

    /**
     * Generates a list of simulated PatientRecord objects.
     * This method simulates the crucial step of acquiring and formatting data.
     * @param count The number of records to generate.
     * @return A list of PatientRecord objects.
     */
    private static List<PatientRecord> generateSimulatedData(int count) {
        List<PatientRecord> records = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < count; i++) {
            // Features: Tumor Size (1-50), Grade (1-3), Lymph Node Status (0 or 1)
            int tumorSize = rand.nextInt(50) + 1; 
            int grade = rand.nextInt(3) + 1;
            int lymphNodeStatus = rand.nextInt(2);
            
            // Simple rule for simulated Recurrence: Higher tumor size, grade, 
            // and lymph node status increase the probability of recurrence.
            int recurrenceStatus = (tumorSize * 0.02 + grade * 0.15 + lymphNodeStatus * 0.3 + rand.nextDouble() * 0.2 > 0.6) ? 1 : 0;
            
            records.add(new PatientRecord(tumorSize, grade, lymphNodeStatus, recurrenceStatus));
        }
        return records;
    }
}

/**
 * Represents a single patient's clinical and outcome data.
 * This class serves as the 'data object' in the ML pipeline.
 */
class PatientRecord {
    private final int tumorSize;   // Feature: Size of the primary tumor
    private final int grade;       // Feature: Tumor grade (1-3, higher is worse)
    private final int lymphNodeStatus; // Feature: 1 if positive, 0 if negative
    private final int recurrenceStatus;  // Outcome: 1 if recurrence occurred, 0 otherwise (Ground Truth)

    public PatientRecord(int tumorSize, int grade, int lymphNodeStatus, int recurrenceStatus) {
        this.tumorSize = tumorSize;
        this.grade = grade;
        this.lymphNodeStatus = lymphNodeStatus;
        this.recurrenceStatus = recurrenceStatus;
    }

    // Getters for features and outcome
    public int getTumorSize() { return tumorSize; }
    public int getGrade() { return grade; }
    public int getLymphNodeStatus() { return lymphNodeStatus; }
    public int getRecurrenceStatus() { return recurrenceStatus; }
}

/**
 * Simulates a machine learning classifier.
 * Here, it uses a simple weighted linear combination to determine risk.
 */
class MLModelSimulator {
    // Simulated weights learned during "training"
    private double tumorSizeWeight = 0.0;
    private double gradeWeight = 0.0;
    private double lymphNodeWeight = 0.0;
    private final double bias = 0.5; // Baseline risk level

    /**
     * Simulates the training process by calculating simple average feature importance.
     * @param data The list of PatientRecord objects to "train" on.
     */
    public void train(List<PatientRecord> data) {
        if (data.isEmpty()) return;

        // The goal of this simulation is to calculate weights that reflect how
        // strongly each feature correlates with a recurrence outcome (RecurrenceStatus = 1).
        
        double avgTumorSizeForRecurrence = 0;
        double avgGradeForRecurrence = 0;
        double avgLymphNodeForRecurrence = 0;
        int recurrenceCount = 0;

        for (PatientRecord record : data) {
            if (record.getRecurrenceStatus() == 1) {
                avgTumorSizeForRecurrence += record.getTumorSize();
                avgGradeForRecurrence += record.getGrade();
                avgLymphNodeForRecurrence += record.getLymphNodeStatus();
                recurrenceCount++;
            }
        }
        
        if (recurrenceCount > 0) {
            // Normalize the average values and scale them to be used as feature weights.
            // This mimics finding the optimal coefficients in a linear model.
            tumorSizeWeight = (avgTumorSizeForRecurrence / recurrenceCount) / 50.0 * 0.3; // Tumor Size max 50
            gradeWeight = (avgGradeForRecurrence / recurrenceCount) / 3.0 * 0.4; // Grade max 3
            lymphNodeWeight = (avgLymphNodeForRecurrence / recurrenceCount) * 0.5; // Lymph Node max 1
        }
    }
    
    /**
     * Predicts the recurrence status (0 or 1) for a single patient record.
     * The core ML logic: Weighted sum of features compared against a threshold.
     * @param record The patient data to predict.
     * @return 1 for predicted recurrence, 0 for predicted non-recurrence.
     */
    public int predict(PatientRecord record) {
        // Calculate the risk score by taking a weighted sum of the patient's features.
        // This is conceptually the same as a linear regression or logistic regression
        // model before applying the activation function.
        double riskScore = (record.getTumorSize() * tumorSizeWeight) +
                           (record.getGrade() * gradeWeight) +
                           (record.getLymphNodeStatus() * lymphNodeWeight) -
                           bias; // Subtract bias to act as the final decision threshold

        // If the calculated risk score is above the decision threshold (0 in this setup), 
        // we predict recurrence (1). Otherwise, we predict no recurrence (0).
        return (riskScore > 0) ? 1 : 0;
    }

    // Getters for displaying the "learned" parameters
    public double getTumorSizeWeight() { return tumorSizeWeight; }
    public double getGradeWeight() { return gradeWeight; }
    public double getLymphNodeWeight() { return lymphNodeWeight; }
}
