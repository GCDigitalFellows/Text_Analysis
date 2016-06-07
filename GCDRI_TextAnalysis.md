
# Introduction to Text Analysis with Python
<br><br>
### GCDRI June 8, 2016
### Michelle McSweeney 
### @MMcSweeney7
### \#GCDRI

Major steps in doing text or language analysis
1. Getting the data
2. **Turning the data into numbers**
3. Analyzing the data 

### Question

What parts of language (spoken, written, texted) can you count? 

What numbers can you come up with beyond counting?

Text Editor Functions

* Word counts
* Character Counts
* Word-in-context (text wrangler & sublime)

Moving Beyond the Text Editor with Python + NLTK

The Natural Language ToolKit


```python
import nltk
from nltk.book import *
```

    *** Introductory Examples for the NLTK Book ***
    Loading text1, ..., text9 and sent1, ..., sent9
    Type the name of the text or sentence to view it.
    Type: 'texts()' or 'sents()' to list the materials.
    text1: Moby Dick by Herman Melville 1851
    text2: Sense and Sensibility by Jane Austen 1811
    text3: The Book of Genesis
    text4: Inaugural Address Corpus
    text5: Chat Corpus
    text6: Monty Python and the Holy Grail
    text7: Wall Street Journal
    text8: Personals Corpus
    text9: The Man Who Was Thursday by G . K . Chesterton 1908


**Concordance**: Words in their contexts


```python
text1.concordance("love")
```

**Similar**: Words that appear in a similar environment as the target word

Might do this with two different texts


```python
text1.similar("hope")
```


```python
text2.similar("hope")
```

**Common Contexts**: Two words that appear in similar environments as each other


```python
text1.common_contexts(['whale','monster'])
```


```python
text1.dispersion_plot(["whale", "monster"])
```

The Dispersion Plot is usually **BEHIND** the browser!

**CLOSE THE DISPERSION PLOT BEFORE WE MOVE ON**


```python
Now we are going to calculate a bunch of things related to my text without NLTK - just using Python
```

Count a specific word - how many times does this sequence of characters occur in my document?


```python
text1.count("love")
```

How many **tokens** are in my text?

**tokens** are unique sequences, let's start with an example:

"love", "bowie", "Bowie", "!" and ":)" are all unique **tokens**


```python
len(text1)
```

How many **unique** words are in my text? 

* first make a set that groups all the "words" together (numbers, punctuation sequences, etc.) - this groups together **types**. 
* Token = instance
* Type = more general ("bowie" and "Bowie" are different types - why?)


```python
set(text1)
```

I like to **sort** my **set** so I know what I have:


```python
sorted(set(text1))
```

Now count the number of items in that set to find the number of unique words


```python
len(set(text1))
```

**Lexical Density**: the number of unique tokens divided by the total number of words. 

This is a descriptive measure of language register or grade level approximations.



```python
len(set(text1))/len(text1)
```




    0.07406285585022564



**Frequency Distribution** is a probability object that Python deals with. We will use it to make a graph of the most common words.


```python
my_dist = FreqDist(text1)
```

Since nothing appears, I like to go check for it


```python
type(my_dist)
```

Now let's **plot** the graph


```python
my_dist.plot(50,cumulative=False)
```


```python
It may be a little easier to look at as a list
```


```python
my_dist.most_common(100)
```

That actually doesn't tell us very much. 

**PREVIEW** <br>
*We need to remove the **stopwords** to learn more.*


```python
love_words = ['love', 'joy', 'hope', 'amor']
```


```python
my_list = []
for word in love_words:
    if word in text8:
        my_list.append(word)
    else:
        pass
```


```python
print(my_list)
```

Let's pull a book in from the Internet

Project Gutenburg is a great source! www.gutenberg.org

Text 37472 is Zanzibar Tales translated by George W. Bateman



To make a book into a Text NLTK can deal with, we have to:

* open the file from a location
* read it/decode it 
* tokenize it (go from a string to a list of word)
* nltk.Text() 


```python
#timport the urlopen commans
from urllib.request import urlopen
#set the url to a variable
my_url = "https://www.gutenberg.org/files/37472/37472.txt"

```


```python
#open the file from the url
file = urlopen(my_url)
#read the opened file
raw = file.read()
```


```python
#specify which decoding to use. (usually utf-8)
zt = raw.decode('utf-8')
```


```python
#check the type to be sure it worked. I expect a string now.
type(zt)
```




    str



If this doesn't work, or you are dealing with messier text, check out the ftfy library
Fixes Text For You   

http://ftfy.readthedocs.io


```python
import ftfy.bad_codecs

bad_zt = raw.decode('utf-8-variants')
```


```python
type(zt)
```




    str




```python
#split the string into words with word_tokenize
zt_tok = nltk.word_tokenize(zt)
```


```python
#check to make sure it worked
type(zt_tok)
```




    list




```python
#get an idea of how big the file is
len(zt_tok)
```




    34860




```python
#look at the first 10 words to be sure its correct
zt_tok[:10]
```




    ['The',
     'Project',
     'Gutenberg',
     'EBook',
     'of',
     'Zanzibar',
     'Tales',
     ',',
     'by',
     'Various']



Yuck! That just looks like metadata! 

Removing metadata involves using Regular Expressions

Regular Expressions are saved for another day. They are powerful but complicated


```python
#I want to get rid of that intro metadata!!

#I found the number 177 by copying this into Text Wrangler
#Text Wrangler counts words, characters, and spaces, which makes this easier

#To do this for many files, you will need Regular Expressions (not covered here)

zt_tok[177:188]
```




    ['TO',
     'MY',
     'READERS',
     '.',
     'Thirty',
     'years',
     'ago',
     'Central',
     'Africa',
     'was',
     'what']




```python
#turn the list of words into a text nltk can recognize
zt_text = nltk.Text(zt_tok[177:])
```


```python
#check to make sure it worked
type(zt_text)
```




    nltk.text.Text




```python
#get an idea of how big the file is
len(zt_text)
```




    34683



One step further! 
**Part-of-Speech Tagging**

NLTK uses the Penn Tag Set. 

There are better options (i.e., tree tagger and polyglot), but this illustrates the idea.


```python
#make a new object that has all the words and tags in it
zt_tagged = nltk.pos_tag(zt_text)
```


```python
print(zt_tagged[:10])
```

    [('TO', 'NNP'), ('MY', 'NNP'), ('READERS', 'NNP'), ('.', '.'), ('Thirty', 'CD'), ('years', 'NNS'), ('ago', 'RB'), ('Central', 'NNP'), ('Africa', 'NNP'), ('was', 'VBD')]


This doesn't look like a dictionary! What's going on??


```python
type(zt_tagged)
```




    list



We have a **list of tuples**


I'm going to put it in a for-loop

Have to deal with this in a special way (a, b) in my_list



```python
#function to determine what is the most common tag in Zanzibar Tales
def commontag(taggedbook):
#create an empty dictionary
    tag_dict = {}
#for every word/tag combo in my list, 
    for (word, tag) in taggedbook:
        if tag in tag_dict: 
            tag_dict[tag]+=1
        else:
            tag_dict[tag] = 1
    print(tag_dict)

commontag(zt_tagged)
```

    OrderedDict([('$', 2), ('WP$', 6), ('RBR', 7), ('RBS', 8), ('FW', 8), ('NNPS', 11), (')', 22), ('(', 22), ('JJS', 26), ('JJR', 28), ('PDT', 46), ('EX', 68), ('UH', 72), ('POS', 109), ('WDT', 146), ('RP', 209), ('WP', 211), ('CD', 290), ('WRB', 297), ('MD', 462), ('VBZ', 467), ('VBG', 493), (':', 571), ('VBN', 591), ('PRP$', 632), ('``', 719), ('VBP', 739), ('TO', 794), ("''", 837), ('NNS', 885), ('JJ', 1297), ('NNP', 1328), ('VB', 1386), ('CC', 1461), ('.', 1605), ('RB', 1632), ('VBD', 2171), (',', 2639), ('IN', 2731), ('DT', 2941), ('PRP', 3000), ('NN', 3714)])


I wish I would have made that return an ordered dictionary!!!

Let's add a line of code with OrderedDict in it'


```python
from collections import OrderedDict

def commontag(taggedbook):
#create an empty dictionary
    tag_dict = {}
#for every word/tag combo in my list, 
    for (word, tag) in taggedbook:
        if tag in tag_dict: 
            tag_dict[tag]+=1
        else:
            tag_dict[tag] = 1
    tag_dict = OrderedDict(sorted(tag_dict.items(), key=lambda t: t[1]))
    print(tag_dict)

commontag(zt_tagged)
```

How do you know to put all that other stuff in (i.e., sorted, lambda, etc)?!?

*Read the docs*

https://docs.python.org/3.1/whatsnew/3.1.html

So far, we have counted things in our texts by looking at 

* Concordance
* Words in similar environments
* Words in common contexts
* Unique words 
* Length of words 


Then we performed some operations, but still counted things:

* Frequency Distributions
* Lexical Density 
* Found words from a list in a text
* Part-of-Speech Tags


```python
Now we will perform operations on the Text itself before doing those operations
```

To get a better idea of the content of a text, it's usually best to exclude stopwords

**Stopwords** perform grammatical functions, but have limited semantic content.

(the, a, at, in, of, with, etc.)

Two methods:

* Use a pre-defined list from nltk
* Make your own list and use a for-loop


```python
from nltk.corpus import stopwords

```


```python
#have to tell NLTK that you want the English stopwords
mystops = stopwords.words('english')
```


```python
nostop_text1 = []

#remove stop words
#go through all the words in text1 and save all the non-stopwords
for word in text1:
    if word not in mystops:
        nostop_text1.append(word)
    else:
        pass
    

```


```python
print(nostop_text1[50:150])
```

    ['.', 'He', 'loved', 'dust', 'old', 'grammars', ';', 'somehow', 'mildly', 'reminded', 'mortality', '.', '"', 'While', 'take', 'hand', 'school', 'others', ',', 'teach', 'name', 'whale', '-', 'fish', 'called', 'tongue', 'leaving', ',', 'ignorance', ',', 'letter', 'H', ',', 'almost', 'alone', 'maketh', 'signification', 'word', ',', 'deliver', 'true', '."', '--', 'HACKLUYT', '"', 'WHALE', '.', '...', 'Sw', '.', 'Dan', '.', 'HVAL', '.', 'This', 'animal', 'named', 'roundness', 'rolling', ';', 'Dan', '.', 'HVALT', 'arched', 'vaulted', '."', '--', 'WEBSTER', "'", 'S', 'DICTIONARY', '"', 'WHALE', '.', '...', 'It', 'immediately', 'Dut', '.', 'Ger', '.', 'WALLEN', ';', 'A', '.', 'S', '.', 'WALW', '-', 'IAN', ',', 'roll', ',', 'wallow', '."', '--', 'RICHARDSON', "'", 'S', 'DICTIONARY']


Now to:
* remove all that punctuation 
* make everything lowercase


```python
#remove punctuation
#go through all the items in text1 (without stopwords), and save everything that is alphabetic
nopunct_text1 = []
for word in nostop_text1:
    if word.isalpha():
        nopunct_text1.append(word)
    else:
        pass

```


```python
lower_text1 = []

for w in nopunct_text1:
    lower_text1.append(w.lower())
    
print(lower_text1[:50])
```

    ['moby', 'dick', 'herman', 'melville', 'etymology', 'supplied', 'late', 'consumptive', 'usher', 'grammar', 'school', 'the', 'pale', 'usher', 'threadbare', 'coat', 'heart', 'body', 'brain', 'i', 'see', 'he', 'ever', 'dusting', 'old', 'lexicons', 'grammars', 'queer', 'handkerchief', 'mockingly', 'embellished', 'gay', 'flags', 'known', 'nations', 'world', 'he', 'loved', 'dust', 'old', 'grammars', 'somehow', 'mildly', 'reminded', 'mortality', 'while', 'take', 'hand', 'school', 'others']


There's another way to write this, if this is easier to understand:



```python
new_text1 = [w for w in text1 if w not in mystops]
```

Now you have a nice clean text. 

Let's look at the lexical density of that text


```python
len(set(new_text1))/len(new_text1)
```




    0.11871945561398083



That's much higher!! 

This is the density of the whole book without the stop words, which better represents the variety of words being used

What if I want to read in my OWN corpus?


```python
f = open("/Users/michellejohnson/Desktop/projects/Thailand.txt", 'r')
my_file = f.read()

```


```python
type(my_file)
```




    str



**Going Forward**

* Use a text editor to write complete programs
    * Run these in the terminal
* Use Spyder to write complete programs
* Often save the program you write in the same file as the file you will be working with to shorted the path.

How do I know where to go?!?
* http://www.nltk.org/book_1ed
* http://www.nltk.org/
    
* Come to office hours
* Come to Python Users' Group
* Play! 
    * http://techblog.about.com/post/140231383537/analyzing-the-language-of-the-presidential-debates
    * http://andybromberg.com/sentiment-analysis-python/
    * etc.

