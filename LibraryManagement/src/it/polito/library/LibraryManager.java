package it.polito.library;

import java.util.Map;
import java.util.*;
import java.util.Set;
import java.util.SortedMap;
import java.util.stream.Collectors;


public class LibraryManager {
	
    // R1: Readers and Books   
	
    /**
	 * adds a book to the library archive
	 * The method can be invoked multiple times.
	 * If a book with the same title is already present,
	 * it increases the number of copies available for the book
	 * 
	 * @param title the title of the added book
	 * @return the ID of the book added 
	 */
	SortedMap<String , Integer> bookMap = new TreeMap<>();
	Integer bookId = 999;
	Integer readerId = 1000;
	List<Book>bookList = new ArrayList<>();
	List<Reader>readerList = new ArrayList<>();
	List<Rent>rentList = new ArrayList<>();
	List<EndRental>endRentList = new ArrayList<>();
	
	
	
    public String addBook(String title) {
    	bookId++;
    	String bid = String.valueOf(bookId);
    	Book b = new Book(title, bid);
    	bookList.add(b);
    	
        return String.valueOf(bookId);
    }
    
    /**
	 * Returns the book titles available in the library
	 * sorted alphabetically, each one linked to the
	 * number of copies available for that title.
	 * 
	 * @return a map of the titles liked to the number of available copies
	 */
    public SortedMap<String, Integer> getTitles() {  
    	SortedMap<String, Integer> res = new TreeMap<>();
    	for(Book x: bookList) {
    		if(!res.containsKey(x.name)) {
    			res.put(x.name, 1);
    		}
    		else {
    			Integer count = res.get(x.name);
    			count++;
    			res.put(x.name, count);
    		}
    	}
    	
        return res;
    }
    
    /**
	 * Returns the books available in the library
	 * 
	 * @return a set of the titles liked to the number of available copies
	 */
    public Set<String> getBooks() {
    	Set<String> s = new HashSet<>();
    	for(Book x: bookList) {
    		s.add(String.valueOf(x.id));
    	}
        return s;
    }
    
    /**
	 * Adds a new reader
	 * 
	 * @param name first name of the reader
	 * @param surname last name of the reader
	 */
    public void addReader(String name, String surname) {
    	String rid = String.valueOf(readerId);
    	Reader r = new Reader (name, surname, rid);
    	readerList.add(r);
    	readerId++;
    	System.out.println(readerId);
    	
    }
    
    
    /**
	 * Returns the reader name associated to a unique reader ID
	 * 
	 * @param readerID the unique reader ID
	 * @return the reader name
	 * @throws LibException if the readerID is not present in the archive
	 */
    public String getReaderName(String readerID) throws LibException {
    	boolean flag = false;
    	for (Reader r:readerList) {
    		if(r.id.equals(readerID))
    			flag = true;
    	}
    	if (flag == false)
    		throw new LibException("readerId not found");
    	
    	String nameSur ="";
    	for (Reader r:readerList) {
    		if(r.id.equals(readerID))
    			nameSur = r.name+" " + r.surname;
    			
    	}
    	
        return nameSur;
    }    
    
    
    // R2: Rentals Management
    
    
    /**
	 * Retrieves the bookID of a copy of a book if available
	 * 
	 * @param bookTitle the title of the book
	 * @return the unique book ID of a copy of the book or the message "Not available"
	 * @throws LibException  an exception if the book is not present in the archive
	 */
    public String getAvailableBook(String bookTitle) throws LibException {
    	
    	boolean flagb = bookList.stream().anyMatch(x -> x.name.equals(bookTitle));
    	if (!flagb) 
    		throw new LibException("book isn't in archive");

    	
    	boolean flagA = false;
    	
    	for(Book x: bookList) {
    		if(x.name.equals(bookTitle) && x.rented == false ) 
    			flagA = true;
    	}
    	if(flagA == false)
    		return "Not available";
    	
    	
    	List<String> ids = new ArrayList<>();
    	for(Book x: bookList) {
    		if(x.name.equals(bookTitle) && x.rented == false )
    			ids.add(x.id);
    			
    	}
    	//System.out.println(ids);
    	if(ids.size() == 0 )
    		return "Not Available";
    	return ids.get(0);
    }   

    /**
	 * Starts a rental of a specific book copy for a specific reader
	 * 
	 * @param bookID the unique book ID of the book copy
	 * @param readerID the unique reader ID of the reader
	 * @param startingDate the starting date of the rental
	 * @throws LibException  an exception if the book copy or the reader are not present in the archive,
	 * if the reader is already renting a book, or if the book copy is already rented
	 */
	public void startRental(String bookID, String readerID, String startingDate) throws LibException {
//		boolean flagb = false;
//		for(Book x: bookList) {
//			
//		}

		if (bookList.stream().anyMatch(x -> x.id.equals(bookID) && x.rented))
		    throw new LibException("book already in rent");
		
		boolean flagb = bookList.stream().anyMatch(x -> x.id.equals(bookID));
		if (!flagb) 
			throw new LibException("book isn't in archive");

		boolean flagr = readerList.stream().anyMatch(x -> x.id.equals(readerID));
		if (!flagr) 
			throw new LibException("reader not available");

		if (rentList.stream().anyMatch(x -> x.readerID.equals(readerID) && endRentList.stream().noneMatch(y -> y.readerID.equals(readerID)))) 
			throw new LibException("user with ongoing rental");

		boolean override = rentList.stream().filter(x -> x.readerID.equals(readerID) && x.bookID.equals(bookID))
                .peek(x -> x.startingDate = startingDate).findFirst().isPresent();
		
		if(override == false) {
		Rent r = new Rent(bookID, readerID, startingDate);
	    rentList.add(r);}
	    
	    for(Book x: bookList) {
	    	if(x.id.equals(bookID))
	    		x.rented = true;
	    }
		
		
    }
    
	/**
	 * Ends a rental of a specific book copy for a specific reader
	 * 
	 * @param bookID the unique book ID of the book copy
	 * @param readerID the unique reader ID of the reader
	 * @param endingDate the ending date of the rental
	 * @throws LibException  an exception if the book copy or the reader are not present in the archive,
	 * if the reader is not renting a book, or if the book copy is not rented
	 */
    public void endRental(String bookID, String readerID, String endingDate) throws LibException {
    	if (bookList.stream().anyMatch(x -> x.id.equals(bookID) && !x.rented)) 
    	    throw new LibException("book is not rented");
    	
    	if (readerList.stream().noneMatch(x -> x.id.equals(readerID))) 
    	    throw new LibException("reader not available");
    	
    	if (bookList.stream().noneMatch(x -> x.id.equals(bookID))) 
    	    throw new LibException("book isn't in archive");
    	
    	EndRental e = new EndRental(bookID, readerID, endingDate);
    	endRentList.add(e);
    	
    	bookList.stream().filter(x -> x.id.equals(bookID)).forEach(x -> x.rented = false);
    	
    }
    
    
   /**
	* Retrieves the list of readers that rented a specific book.
	* It takes a unique book ID as input, and returns the readers' reader IDs and the starting and ending dates of each rental
	* 
	* @param bookID the unique book ID of the book copy
	* @return the map linking reader IDs with rentals starting and ending dates
	* @throws LibException  an exception if the book copy or the reader are not present in the archive,
	* if the reader is not renting a book, or if the book copy is not rented
	*/
    public SortedMap<String, String> getRentals(String bookID) throws LibException {
   	
    	boolean flagb = bookList.stream().anyMatch(x -> x.id.equals(bookID));
    	if (!flagb) 
    		throw new LibException("book isn't in archive");
    		
    	SortedMap<String, String> rentals = new TreeMap<>();
    	for(Rent x: rentList) {
    		
    		if(x.bookID == bookID)
    			rentals.put(x.readerID,x.startingDate+ " "+ "ONGOING" );
    		
    	}
    	
    	for(EndRental y: endRentList) {
    		if(rentals.containsKey(y.readerID)) {
    			String val = rentals.get(y.readerID);
    			String val2 = val.split(" ")[0];
    			String val3 = val2 + " "+ y.endingDate; 
    			rentals.put(y.readerID, val3);
    			}
    		
    		
    	}
    	
        return rentals;
    }
    
    
    // R3: Book Donations
    
    /**
	* Collects books donated to the library.
	* 
	* @param donatedTitles It takes in input book titles in the format "First title,Second title"
	*/
    public void receiveDonation(String donatedTitles) {
    	String[] titles = donatedTitles.split(",");
    	for(String x: titles) {
    		addBook(x);
    	}
    	
    }
    
    // R4: Archive Management

    /**
	* Retrieves all the active rentals.
	* 
	* @return the map linking reader IDs with their active rentals

	*/
    public Map<String, String> getOngoingRentals() {
    	Map<String, String> ongRental = new HashMap<>();
    	rentList.stream().filter(x -> endRentList.stream().noneMatch(y -> y.bookID.equals(x.bookID) && y.readerID.equals(x.readerID)))
        .forEach(x -> ongRental.put(x.readerID, x.bookID));
        return ongRental;
    }
    
    /**
	* Removes from the archives all book copies, independently of the title, that were never rented.
	* 
	*/
    public void removeBooks() {
    	List<Book> removeBook = bookList.stream().filter(x -> rentList.stream().noneMatch(y -> x.id.equals(y.bookID)))
    	        .collect(Collectors.toList());
    	bookList.removeAll(removeBook);
    }
    	
    // R5: Stats
    
    /**
	* Finds the reader with the highest number of rentals
	* and returns their unique ID.
	* 
	* @return the uniqueID of the reader with the highest number of rentals
	*/
    public String findBookWorm() {
    	
    	String maxKey = rentList.stream().collect(Collectors.groupingBy(rent -> rent.readerID,TreeMap::new,Collectors.counting()))
    	        .entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    	return maxKey;
    }
    
    /**
	* Returns the total number of rentals by title. 
	* 
	* @return the map linking a title with the number of rentals
	*/
    public Map<String,Integer> rentalCounts() {

    	Map<String, Integer> rentCount = bookList.stream()
    	        .collect(Collectors.toMap(
    	                book -> book.name,
    	                book -> (int) rentList.stream().filter(rent -> rent.bookID.equals(book.id)).count(),
    	                Integer::sum,
    	                HashMap::new
    	        ));

    	return rentCount;

        
    }

}

