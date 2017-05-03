package cz.mendelu.nsa;

import com.sun.org.apache.xpath.internal.SourceTree;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.CharArrayReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class Main {


    public static List<String> fontPaths = new ArrayList<>();

    public static void main(String[] args) {

        List<String> captchas = new ArrayList<>();
        captchas.add("hello nsa project");
        captchas.add("i love apple brand");
        captchas.add("manageiq");
        captchas.add("legless lego legolass");
        captchas.add("leglessness");
        captchas.add("alphabet");
        captchas.add("abcdefghijklmnopqrstuvwxyz");
        captchas.add("virtual machine");
        captchas.add("fedora");
        captchas.add("rip in peace");
        captchas.add("love");
        captchas.add("mendel university in brno");
        captchas.add("red alert two");
        captchas.add("zerglins at the gates");

        try {
            // pokud jde o volání z Pythonu
            if (args.length > 0) {
                argsCall(args);
                System.exit(0);
            }

//        generování jen jedním fontem
            fontPaths.add("NSA_project/resources/fonts/boxpot.ttf");

            // generování více fonty
//            fontPaths.addAll(loadFonts());

//             font settings in "generate" method
            generateAlphabet("NSA_project/resources/output/alphabet_9", 1000);



            int i = 0;
            for (String captcha : captchas) {
                generate(captcha, "NSA_project/resources/output/alphabet_9/" + "captcha" + i + ".png");
                i++;
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static void generate(String str, String outputFile) {

        String fontPath = fontPaths.get(new Random().nextInt(fontPaths.size()));
//        String fontPath = "../resources/fonts/boxpot.ttf";

        CaptchaGenerator generator = new CaptchaGenerator(str.length() * 32, 32, fontPath, 30, Boolean.TRUE, 10, 30, 40);

        generator.setup();

        String format = outputFile.substring(outputFile.lastIndexOf(".") + 1, outputFile.length());

        char[] charArray = str.toCharArray();



        // random uppercase
//        Random rand = new Random();
//        for(int i = 0; i < charArray.length;i++){
//            int randomNum = rand.nextInt(10) + 1;
//            if (randomNum > 5)
//                charArray[i] = Character.toUpperCase(charArray[i]);
//        }


        BufferedImage bi = generator.createCaptcha(charArray);


        bi = addNoise(bi, 2);

        File outfile = new File(outputFile);
        try {
            ImageIO.write(bi, format, outfile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static BufferedImage addNoise(BufferedImage bf, int power) {

        Random rand = new Random();

        Color myWhite = new Color(0, 0, 0); // Color white
        int rgb = myWhite.getRGB();

        for (int i = 0; i < bf.getHeight(); i++) {
            for (int j = 0; j < bf.getWidth(); j++) {
                int randomNum = rand.nextInt(10) + 1;
                if (randomNum > (10 - power))
                    bf.setRGB(j, i, rgb);
            }
        }

        return bf;
    }

    /**
     * Vygeneruje abecedu
     *
     * @param outDir  Zadejte kořenový adresář pri vygenerování abecedy
     * @param samples Počet vzorků na každý znak
     */
    private static void generateAlphabet(String outDir, int samples) {
        Long start = System.currentTimeMillis();
        System.out.println("Generuji abecedu do [/" + outDir + "] počet vzorků na písmeno [" + samples + "]");

        char[] alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray();

        File root = makeDir(outDir);

        for (char c : alphabet) {

            File charFolder = makeDir(root.getPath() + "/" + c);

            for (int i = 0; i < samples + 1; i++) {
                System.out.println("Generuji písmeno [" + c + "] vzorek [" + i + "]");
                generate(Character.toString(c), charFolder.getPath() + "/" + i + ".jpg");
            }
        }

        Long end = System.currentTimeMillis();
        System.out.println("Abeceda vygenerovana za [" + (end - start) / 1000 + " sekund]");


    }


    /**
     * Pro zavolani přes java -jar
     *
     * @param args argumenty pro generování, dle pořadí
     *             0. - řetězec ,který se má vygenerovat
     *             1. - název souboru, který se má vytvořit (včetně přípony)
     *             2. - cesta k souboru s fontem
     *             3. - výška (nepovinné - defaultně 60px)
     *             4. - šířka (nepovinné - defaultně (počet znaků řetězce * 60px))
     *             5. - velikost písma (nepovinné - defaultně 60px)
     *             6. - aplituda rotace (nepovinné - defaultně 30)
     *             7. - aplituda změnvy velikosto (nepovinné - defaultně 40)
     *             8. - vykreslit mřížku (nepovinné - defaultně False)
     *             9. - velikost mřížky (nepovinné - defaultně 10px)
     */
    private static void argsCall(String[] args) {
        String str = args[0];
        String outputFile = args[1];
        String fontFile = args[2];

        int height = (!args[3].equals("d")) ? Integer.parseInt(args[3]) : 60;
        int width = (!args[4].equals("d")) ? Integer.parseInt(args[4]) : str.length() * 60;
        int fontSize = (!args[5].equals("d")) ? Integer.parseInt(args[5]) : 60;
        int rotationAmplitude = (!args[6].equals("d")) ? Integer.parseInt(args[6]) : 30;
        int scaleAmplitude = (!args[7].equals("d")) ? Integer.parseInt(args[7]) : 40;
        boolean showGrid = (!args[8].equals("d")) && Boolean.parseBoolean(args[8]);
        int gridSize = (!args[9].equals("d")) ? Integer.parseInt(args[9]) : 10;


        String format = outputFile.substring(outputFile.lastIndexOf(".") + 1, outputFile.length());

        CaptchaGenerator generator = new CaptchaGenerator(width, height, fontFile, fontSize, showGrid, gridSize, rotationAmplitude, scaleAmplitude);

        char[] charArray = str.toCharArray();

        generator.setup();
        BufferedImage bi = generator.createCaptcha(charArray);

        File outfile = new File(outputFile);
        try {
            ImageIO.write(bi, format, outfile);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public static File makeDir(String path) {

        File theDir = new File(path);

        if (theDir.exists()) {
            delDir(path);
        }


        if (!theDir.exists()) {
            try {
                theDir.mkdir();
            } catch (SecurityException se) {
                se.printStackTrace();
            }
        }
        return theDir;

    }


    public static void delDir(String path) {
        File f = new File(path);
        if (f.isDirectory()) {
            for (File c : f.listFiles())
                delDir(c.getPath());
        }
        if (!f.delete())
            try {
                throw new FileNotFoundException("Failed to delete file: " + f);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
    }

    public static ArrayList<String> loadFonts() {
        File fontDir = new File("NSA_project/resources/fonts/");

        ArrayList<String> fontPaths = new ArrayList<>();

        File[] listOfFiles = fontDir.listFiles();

        for (File font : listOfFiles) {
            fontPaths.add(font.getPath());
        }

        return fontPaths;
    }


}
