import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
public class Append {
	public static void main(String args[]) {
		
		
	}


public static ArrayList appendToArray(String path) throws IOException {
		FileReader fr = new FileReader(path);
		BufferedReader bf = new BufferedReader(fr);
		ArrayList<String> array = new ArrayList();
		for(int i = 0; i >= getNumberOfLines(path); i += 2) {
			array.add(bf.read(i));
		}
		
	}
public static int getNumberOfLines(String path) throws IOException {
	FileReader fr = new FileReader(path);
	BufferedReader bf = new BufferedReader(fr);
	int numLines = 0;
	String aLine;
	while((aLine = bf.readLine()) != null) {
		numLines++;
	}
	return numLines;
	}
}
