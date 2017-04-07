package cz.mendelu.nsa;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Random;

/**
 * Modified source code commons-captcha (LGPL licence)
 * Source: https://github.com/shred/commons-captcha
 */
public class CaptchaGenerator {

    private final Random rnd = new Random();

    private int width;

    private int height;

    private String fontPath;

    private float fontSize;

    private boolean showGrid;

    private int gridSize;
    private int rotationAmplitude;
    private int scaleAmplitude;
    private Font font;

    public CaptchaGenerator(int width, int height, String fontPath, float fontSize, boolean showGrid, int gridSize, int rotationAmplitude, int scaleAmplitude) {
        this.width = width;
        this.height = height;
        this.fontPath = fontPath;
        this.fontSize = fontSize;
        this.showGrid = showGrid;
        this.gridSize = gridSize;
        this.rotationAmplitude = rotationAmplitude;
        this.scaleAmplitude = scaleAmplitude;
    }

    public void setShowGrid(boolean showGrid) {
        this.showGrid = showGrid;
    }

    public void setGridSize(int gridSize) {
        this.gridSize = gridSize;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public void setFontPath(String fontPath) {
        this.fontPath = fontPath;
    }

    public void setFontSize(float fontSize) {
        this.fontSize = fontSize;
    }

    public void setRotationAmplitude(int rotationAmplitude) {
        this.rotationAmplitude = rotationAmplitude;
    }

    public void setScaleAmplitude(int scaleAmplitude) {
        this.scaleAmplitude = scaleAmplitude;
    }

    public void setup() {
        if (this.width > 0 && this.height > 0) {
            if (this.fontPath == null) {
                throw new IllegalStateException("Font is not set");
            } else {
                try {
                    InputStream ex = new FileInputStream(fontPath);
                    this.font = Font.createFont(0, ex);
                } catch (Exception var2) {
                    System.err.println("font path - " + this.fontPath );
                    System.out.println("font path - " + this.fontPath );
                    throw new IllegalStateException();
                }
            }
        } else {
            throw new IllegalStateException("Captcha size is not set");
        }
    }

    public BufferedImage createCaptcha(char[] text) {
        if (text != null && text.length != 0) {
            BufferedImage image = new BufferedImage(this.width, this.height, 1);
            Graphics2D g2d = image.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setBackground(Color.WHITE);
            g2d.setColor(Color.BLACK);
            this.clearCanvas(g2d);
            if (this.showGrid) {
                this.drawGrid(g2d);
            }

            int charMaxWidth = this.width / text.length;
            int xPos = 0;
            int len$ = text.length;

            for (int i$ = 0; i$ < len$; ++i$) {
                char ch = text[i$];
                this.drawCharacter(g2d, ch, xPos, charMaxWidth);
                xPos += charMaxWidth;
            }

            g2d.dispose();
            return image;
        } else {
            throw new IllegalArgumentException("No captcha text given");
        }
    }

    private void clearCanvas(Graphics2D g2d) {
        g2d.clearRect(0, 0, this.width, this.height);
    }

    private void drawGrid(Graphics2D g2d) {
        int x;
        for (x = 2; x < this.height; x += this.gridSize) {
            g2d.drawLine(0, x, this.width - 1, x);
        }
        for (x = 2; x < this.width; x += this.gridSize) {
            g2d.drawLine(x, 0, x, this.height - 1);
        }

    }

    private void drawCharacter(Graphics2D g2d, char ch, int x, int boxWidth) {
        double degree = this.rnd.nextDouble() * (double) this.rotationAmplitude * 2.0D - (double) this.rotationAmplitude;
        double scale = 1.0D - this.rnd.nextDouble() * (double) this.scaleAmplitude / 100.0D;
        Graphics2D cg2d = (Graphics2D) g2d.create();
        cg2d.setFont(this.font.deriveFont(this.fontSize));
        cg2d.translate(x + boxWidth / 2, this.height / 2);
        cg2d.rotate(degree * 3.141592653589793D / 90.0D);
        cg2d.scale(scale, scale);
        FontMetrics fm = cg2d.getFontMetrics();
        int charWidth = fm.charWidth(ch);
        int charHeight = fm.getAscent() + fm.getDescent();
        cg2d.drawString(String.valueOf(ch), -(charWidth / 2), fm.getAscent() - charHeight / 2);
        cg2d.dispose();
    }

}
