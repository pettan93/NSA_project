package cz.mendelu.nsa;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class Main {

    public static void main(String[] args) {

        String str = args[0];
        String outputFile = args[1];
        String fontFile = args[2];

        int height = (!args[3].equals("d"))?  Integer.parseInt(args[3]) : 60;
        int width = (!args[4].equals("d"))? Integer.parseInt(args[4]) : str.length() * 60;
        int fontSize = (!args[5].equals("d"))? Integer.parseInt(args[5]) : 60;
        int rotationAmplitude = (!args[6].equals("d"))? Integer.parseInt(args[6]) : 30;
        int scaleAmplitude = (!args[7].equals("d"))? Integer.parseInt(args[7]) : 40;
        boolean showGrid = (!args[8].equals("d")) && Boolean.parseBoolean(args[8]);
        int gridSize = (!args[9].equals("d"))? Integer.parseInt(args[9]) : 10;


        String format = outputFile.substring(outputFile.lastIndexOf(".") + 1, outputFile.length());

        CaptchaGenerator generator = new CaptchaGenerator(width,height,fontFile,fontSize,showGrid,gridSize,rotationAmplitude,scaleAmplitude);

        char[] charArray = str.toCharArray();

        generator.setup();
        BufferedImage bi = generator.createCaptcha(charArray);

        File outputfile = new File(outputFile);
        try {
            ImageIO.write(bi, format, outputfile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
