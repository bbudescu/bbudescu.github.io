---
layout: page
title: About
permalink: /about/
---

# Definition
I am a machine learning and computer vision engineer. I help companies big and small explore the way in which computers can understand things and help people.

# Brief Overview

# The Story So Far
Banal consecution of events: student -> intern -> employee -> freelancer -> manager
Check out my [LinkedIn profile](https://www.linkedin.com/in/bbudescu/) for specifics.

# Tech Specs

Time and again, potential customers have asked me to tell give them a more detailed explanation of the skills I've earned over time. I got tired of writing e-mails, so now I'll just list them here, so that anybody that is interested of the specific technologies I've used can have an in-depth outline.

This section is kind of lengthy and, hence, boring, so feel free to skip it.

In terms of the of generality / specialization (and, to some degree, of chronology), this is a **bottom-up** presentation of my skills. For a top-down approach, please read the following sub-sections in reverse order.

## Computer Programming
The first time I got paid for the code I wrote was in July 2008 and that's how I've earned my living ever since.

Ok, there was a 1-year long break (2009-2010), during which I only wrote the code for a website, but I was still a student back then and had a job as a computer hardware sales/repair-rep, so don't blame me for it.

## C++:
Between July 2010 and August 2016, C++ was the language in which I spent most of the time writing code. Since then, I only use it occasionally, so one could say that my skills are a bit rusty. However, to my surprise, at the time of this writing (October 2019), when having to write a samll C++ project for one of my clients, I discovered that I still remember enough of the tricks I used to know so as to produce a decent quality code.

During the various projects I've taken part to I've had the chance to write both low level code for high performance applications and / or hardware platforms with limited resources and high level code shared with lots of other developers, who needed to understand it easily, so that they can modify, use and extend it. 

Some of the tech I've used with C++:
- stuff in C++'s `STL`, including stuff defined by the C++ 11 / 14 standars (I didn't get to C++ 17, though):
    - containers: `array`, `map`, `set`, `unordered_map`, `unordered_set`, `vector`, and the lesser known (and widely underapreciated, despite haveing a very efficient `IPP`-based implementation when used compiled with Intel's compiler) `valarray`; I'm also a fan of the Oxford comma
    - algorithms: functions the like of which you would find in `<algorithm>` and `<numeric>`. Very interestingly, the `thrust` template library bundled within the CUDA toolkit exposes a very similar API, but is also able to generate CUDA kernels and / or multi-threaded implementations for the operations (under the hood, they would use stuff like `OpenMP` or Intel's Threading Building Blocks (`TBB`) to parallelize the code across available hardware).
    - functional programming elements like functors, lambdas, stuff you would find in `<fucntional>`, especially useful in conjunction with `<algorithm>`, as they allow you to expose parallelism opportunities to the runtime / compiler / framework etc.
    - smart pointers: mainly the `unique_ptr` and `shared_ptr`, but also `weak_ptr`. Funny story: I've also implemented reference counting myself a couple of times, but would not do it again without relying on the stl implementations.
    - multi-threading: `<thread>` API and associated stuff like atomics, locks, semaphores, condition variables, futures etc. I've also written some multi-threaded code using the Windows API, `pthread` and, at some point, I needed a fair lock, which I couldn't find an implementation of in the `STL` at the time and used the one in `boost` (or `tbb`?) instead
    - other stuff, like `<chrono>` and `<random>`
- Boost libraries:
    - most of the functionality I used to use has now been (or is in the process of being) standardized in C++11/14/17/..., so you get it in most `stl` implementations bundled with compilers
    - `Random`: the mersenne-twister implementation in boost was way better then the one in the stl packaged with the C++ compiler delivered with Visual Studio 2015. This actually made a whole bunch of difference when training some neural nets.
    - `Bimap`, `Filesystem`, `Python`, and `Test`
- I remember being able to define type traits and other crazy template stuff (e.g., I found myself using the CRTP at some point), I used to understand move semantics and rvalue references and other things I can't remember right now/
- Google `Test` library
- BLAS API (Intel's MKL, OpenBLAS)

# Details





**email:** `bogdan[at]iddo.ro`
