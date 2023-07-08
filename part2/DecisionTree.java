package part2;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DecisionTree{
    int numCategories;
    int numAtts;
    Set<String> categoryNames;
    List<String> attNames;
    List<Instance> allInstances;
    Node root;

    public DecisionTree(String trainingFileName, String testFileName) {

        //read data files using the helper code - populates fields
        readDataFile(trainingFileName);

        //train the decision tree
        root = buildTree(new HashSet<>(allInstances), attNames);
        System.out.println("Trained tree:");
        printTree(root, 0, "");
        
        //calculate baseline category
        String baseline = mostCommonCategoryNode(allInstances).getName();
        System.out.println("Baseline category: " + baseline);

        //test the decision tree
        readDataFile(testFileName);
        List<String> predictedCategories = new ArrayList<>();
        List<String> actualCategories = new ArrayList<>();
        for(Instance instance : allInstances){
            predictedCategories.add(predictCategory(instance, root));
            actualCategories.add(instance.getCategory());
        }
        System.out.println("Predicted categories: " + predictedCategories);
        System.out.println("Actual categories: " + actualCategories);
        System.out.printf("Accuracy: %.1f%%\n", computeAccuracy(allInstances)*100);

        //accuracy of baseline
        double baselineAccuracy = allInstances.stream()
            .filter(instance -> instance.getCategory().equals(baseline))
            .count() / (double) allInstances.size();
        System.out.printf("Baseline Accuracy: %.1f%%\n", baselineAccuracy*100);

    }

    private double computeAccuracy(List<Instance> instances) {
        double correct = instances.stream()
            .filter(instance -> predictCategory(instance, root).equals(instance.getCategory()))
            .count();
        return correct / instances.size();
    }

    private String predictCategory(Instance instance, Node node) {
        if(node.getTrueSet() == null && node.getFalseSet() == null){
            return node.getName(); //return the category
        }
        int index = attNames.indexOf(node.getName());
        //recursively call the function on the true or false set
        if(instance.getAtt(index)){ 
            return predictCategory(instance, node.getTrueSet());
        }
        return predictCategory(instance, node.getFalseSet());
    }

    private void printTree(Node node, int indent, String decision) {
        if(node == null) return;
        for(int i = 0; i < indent; i++){
            System.out.print("   ");
        }
        String question = "";
        if(node.getTrueSet() == null && node.getFalseSet() == null){
            question = ". probability = "+ node.getProbability();
        } else{
            question = "?";
        }
        System.out.println(decision + node.getName() + question);
        printTree(node.getTrueSet(), indent + 1, "True: ");
        printTree(node.getFalseSet(), indent + 1, "False: ");

    }

    private Node buildTree(Set<Instance> instances, List<String> attributes) {
        if(instances.isEmpty()){
            //System.out.println("empty case!! ");
            return mostCommonCategoryNode(allInstances); //empty case
        } else if(instancesSameClass(instances)){
            return new Node(instances.iterator().next().getCategory(), null, null, 1.0); //pure case
        } else if(attributes.isEmpty()){
            //System.out.println("no attributes left case!!");
            return mostCommonCategoryNode(new ArrayList<>(instances)); //no attributes left
        } else{
            Set<Instance> bestTrueSet = null;
            Set<Instance> bestFalseSet = null;
            double bestPurity = 0;
            String bestAttribute = "";

            for(String attribute : attributes){
                //separate instances into two instances: true or false
                Set<Instance> trueSet = instances.stream()
                    .filter(x -> x.getAtt(attNames.indexOf(attribute)) == true)
                    .collect(Collectors.toSet());
                Set<Instance> falseSet = instances.stream()
                    .filter(x -> x.getAtt(attNames.indexOf(attribute)) == false)
                    .collect(Collectors.toSet());
                double purity = computePurity(trueSet, falseSet);
                if(purity > bestPurity){
                    bestPurity = purity;
                    bestTrueSet = trueSet;
                    bestFalseSet = falseSet;
                    bestAttribute = attribute;
                }
            }
            //remove the attribute that was used to split the instances
            List<String> copyAttributes = new ArrayList<>(attributes);
            copyAttributes.remove(bestAttribute);

            //build the tree recursively
            Node trueNode = buildTree(bestTrueSet, copyAttributes);
            Node falseNode = buildTree(bestFalseSet, copyAttributes);
            return new Node(bestAttribute, trueNode, falseNode, 1.0);
        }
    }

    private double computePurity(Set<Instance> trueSet, Set<Instance> falseSet) {
        double parentSetSize = trueSet.size() + falseSet.size();
        
        //find the number of each category in each set (A = category 1, B = category 2)
        double trueACount = occurencesOfOneCategory(trueSet);
        double trueBCount = trueSet.size() - trueACount;
        double falseACount = occurencesOfOneCategory(falseSet);
        double falseBCount = falseSet.size() - falseACount;

        double trueWeight = trueSet.size() / parentSetSize;
        double falseWeight = falseSet.size() / parentSetSize;
        
        double truePBI = trueSet.isEmpty() ? 0 : (trueACount / trueSet.size()) * (trueBCount / trueSet.size());
        double falsePBI = falseSet.isEmpty() ? 0 : (falseACount / falseSet.size()) * (falseBCount / falseSet.size());

        double impurity = (trueWeight * truePBI + falseWeight * falsePBI); // divde by 2 to get the gini index?
        return 1 - impurity;
    }

    private double occurencesOfOneCategory(Set<Instance> set) {
        String categoryA = categoryNames.iterator().next(); //get the a category to use as the reference
        return set.stream()
                .filter(i -> i.getCategory().equals(categoryA))
                .count();
    }

    private boolean instancesSameClass(Set<Instance> instances) {
        //return true if all instances have the same category
        String category = instances.iterator().next().getCategory();
        return instances.stream().allMatch(i -> i.getCategory().equals(category));
    }

    private Node mostCommonCategoryNode(List<Instance> instances) {
        Map<String, Long> mapOfCategories = instances.stream()
                .map(i -> i.category)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting())); // Map of category -> count
        String name = mapOfCategories 
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
        double probability = (mapOfCategories
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getValue)
                .orElse(null)) / allInstances.size();
        return new Node(name, null, null, probability);
    }

    

    public static void main(String[] args){
        if(args.length != 2){
            System.out.println("Usage: java DecisionTree hepatitis-training hepatitis-test");
            System.exit(1);
        }

        String trainingFileName = args[0];
        String testFileName = args[1];

        new DecisionTree(trainingFileName, testFileName);
        
    }

    private void readDataFile(String fname) {
        /* format of names file:
         * names of attributes (the first one should be the class/category)
         * category followed by true's and false's for each instance
         */
        System.out.println("Reading data from file " + fname);
        try {
            Scanner din = new Scanner(new File(fname));

            attNames = new ArrayList<>();
            Scanner s = new Scanner(din.nextLine());
            // Skip the "Class" attribute.
            s.next();
            while (s.hasNext()) {
                attNames.add(s.next());
            }
            numAtts = attNames.size();
            //System.out.println(numAtts + " attributes");

            allInstances = readInstances(din);
            din.close();

            categoryNames = new HashSet<>();
            for (Instance i : allInstances) {
                categoryNames.add(i.category);
            }
            numCategories = categoryNames.size();
        } catch (IOException e) {
            throw new RuntimeException("Data File caused IO exception");
        }
    }

    private List<Instance> readInstances(Scanner din) {
        /* instance = classname and space separated attribute values */
        List<Instance> instances = new ArrayList<>();
        while (din.hasNext()) {
            Scanner line = new Scanner(din.nextLine());
            instances.add(new Instance(line.next(), line));
        }
        //System.out.println("Read " + instances.size() + " instances");
        return instances;
    }

    private class Node {
        private String name; // can be an attribute or category
        private Node trueSet;
        private Node falseSet;
        private double probability;

        public Node(String name, Node trueSet, Node falseSet, double probability) {
            this.name = name;
            this.trueSet = trueSet;
            this.falseSet = falseSet;
            this.probability = probability;
        }

        public String getName() {
            return name;
        }

        public Node getTrueSet() {
            return trueSet;
        }

        public Node getFalseSet() {
            return falseSet;
        }

        public double getProbability() {
            return probability;
        }
    }
    //copied helper code from helper-code.java into this file for convenience
    private static class Instance {

        private final String category;
        private final List<Boolean> vals;

        public Instance(String cat, Scanner s) {
            category = cat;
            vals = new ArrayList<>();
            while (s.hasNextBoolean()) {
                vals.add(s.nextBoolean());
            }
        }

        public boolean getAtt(int index) {
            return vals.get(index);
        }

        public String getCategory() {
            return category;
        }

        public String toString() {
            StringBuilder ans = new StringBuilder(category);
            ans.append(" ");
            for (Boolean val : vals) {
                ans.append(val ? "true " : "false ");
            }
            return ans.toString();
        }

    }

}