/*
 * Test File class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1998-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/misc/src/testFile.cc,v 1.7 2012/07/11 22:07:58 stolcke Exp $";
#endif

#include <stdlib.h>
#include <string.h>

#include "File.h"

int hasNL(const char *line)
{
	unsigned len = std::strlen(line);

	if (len > 0 && line[len-1] == '\n') {
		return 1;
	} else {
		return 0;
	}
}

int
main()
{
	File file(stdin);

	File buffer("", (size_t)0);

	char *line;

	std::cout << "=== input data ===\n";

	while ((line = file.getline())) {
		file.position(std::cout) << line;

		if (!hasNL(line)) {
			std::cout << "(MISSING NEWLINE)\n";
		}

		// save the line in our buffer
		buffer.fputs(line);
	}

	buffer.fputs("LINE WITHOUT NEWLINE");

	std::cout << "=== buffer contents ===\n";

	unsigned len = std::strlen(buffer.c_str());
	std::cout << "(length = " << len << ")\n";
	std::cout << buffer.c_str();

	std::cout << "\n=== buffer read back ===\n";

	File sfile(buffer.c_str(), len);

	while ((line = sfile.getline())) {
		sfile.position(std::cout) << line;

		if (!hasNL(line)) {
			std::cout << "(MISSING NEWLINE)\n";
		}
	}

	exit(0);
}

