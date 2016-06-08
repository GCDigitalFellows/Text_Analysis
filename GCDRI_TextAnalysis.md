
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


```python
#split the string into words with word_tokenize
zt_tok = nltk.word_tokenize(zt)
```


```python
#check to make sure it worked
type(zt_tok)
```


```python
#get an idea of how big the file is
len(zt_tok)
```


```python
#look at the first 10 words to be sure its correct
zt_tok[:10]
```

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


```python
#turn the list of words into a text nltk can recognize
zt_text = nltk.Text(zt_tok[177:])
```


```python
#check to make sure it worked
type(zt_text)
```


```python
#get an idea of how big the file is
len(zt_text)
```

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

This doesn't look like a dictionary! What's going on??


```python
type(zt_tagged)
```

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

There's another way to write this, if this is easier to understand:



```python
new_text1 = [w for w in text1 if w not in mystops]
```

Now you have a nice clean text. 

Let's look at the lexical density of that text


```python
len(set(new_text1))/len(new_text1)
```

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



```python

```
