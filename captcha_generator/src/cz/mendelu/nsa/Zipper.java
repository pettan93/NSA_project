package cz.mendelu.nsa;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Created by Petr on 09.05.2017.
 */
public class Zipper {


    private String source_folder;

    public Zipper(String source_folder) {
        this.source_folder = source_folder;
    }

    /**
     * Mads Hansen
     * http://stackoverflow.com/questions/15968883/how-to-zip-a-folder-itself-using-java
     */
    public void pack(String zipFilePath) throws IOException {
        Path p = Files.createFile(Paths.get(zipFilePath));
        try (ZipOutputStream zs = new ZipOutputStream(Files.newOutputStream(p))) {
            Path pp = Paths.get(source_folder);
            Files.walk(pp)
                    .filter(path -> !Files.isDirectory(path))
                    .forEach(path -> {
                        if (!path.toString().contains("zip")){
                            ZipEntry zipEntry = new ZipEntry(pp.relativize(path).toString());
                            try {
                                zs.putNextEntry(zipEntry);
                                zs.write(Files.readAllBytes(path));
                                zs.closeEntry();
                            } catch (Exception e) {
                                System.err.println(e);
                            }
                        }
                    });
        }
    }

}
