# Vasp.pm
# Henkelman group Perl module for handling VASP files
# https://github.com/henkelman-group/vasp_scripts

package Vasp;
require Exporter;
@ISA = qw(Exporter);
@EXPORT = qw(read_poscar write_poscar pbc_difference dirkar kardir);

use strict;

sub read_poscar {
    my ($file) = @_;
    open(POS, "<$file") || die "can't open $file: $!\n";
    my $description = <POS>;
    my $scale = <POS>;
    chomp($scale);
    my @basis;
    for (1 .. 3) {
        my $line = <POS>;
        my @v = split(/\s+/, $line);
        shift @v if ($v[0] eq '');
        push @basis, \@v;
    }

    my $atoms_line = <POS>;
    my @atoms = split(/\s+/, $atoms_line);
    shift @atoms if ($atoms[0] eq '');

    my $natoms_line = <POS>;
    my @natoms = split(/\s+/, $natoms_line);
    shift @natoms if ($natoms[0] eq '');

    my $coord_type = <POS>;
    chomp($coord_type);
    my $selectiveflag = 0;
    my @selective;

    if ($coord_type =~ /^[sS]/) {
        $selectiveflag = 1;
        $coord_type = <POS>;
        chomp($coord_type);
    }

    my @coo;
    my $total = 0;
    $total += $_ for @natoms;
    for (1 .. $total) {
        my $line = <POS>;
        my @v = split(/\s+/, $line);
        shift @v if ($v[0] eq '');
        push @coo, [@v[0..2]];
        push @selective, [@v[3..5]] if $selectiveflag;
    }

    close(POS);
    return (\@coo, \@basis, $scale, \@natoms, $total, $selectiveflag, \@selective, $description, "poscar");
}

sub write_poscar {
    my ($coo, $basis, $scale, $natoms, $total, $selectiveflag, $selective, $description, $filename, $filetype) = @_;
    open(POS, ">$filename") || die "can't write to $filename: $!\n";
    print POS $description;
    print POS $scale, "\n";
    for my $b (@$basis) {
        printf POS "  %22.16f %22.16f %22.16f\n", @$b;
    }
    print POS "  ", join("  ", @$natoms), "\n";
    print POS $selectiveflag ? "Selective dynamics\n" : "";
    print POS "Direct\n";
    for my $i (0 .. $#$coo) {
        printf POS "  %22.16f %22.16f %22.16f", @{$coo->[$i]};
        print POS sprintf("  %2s %2s %2s", @{$selective->[$i]}) if $selectiveflag;
        print POS "\n";
    }
    close(POS);
}

sub pbc_difference {
    my ($b, $a, $n) = @_;
    my @diff;
    for my $i (0 .. $n - 1) {
        my @tmp;
        for my $j (0 .. 2) {
            my $d = $b->[$i][$j] - $a->[$i][$j];
            $d -= 1.0 if $d > 0.5;
            $d += 1.0 if $d < -0.5;
            push @tmp, $d;
        }
        push @diff, \@tmp;
    }
    return \@diff;
}

sub dirkar {
    my ($coo, $basis, $scale, $n) = @_;
    for my $i (0 .. $n - 1) {
        my @r = (0, 0, 0);
        for my $j (0 .. 2) {
            for my $k (0 .. 2) {
                $r[$j] += $coo->[$i][$k] * $basis->[$k][$j];
            }
        }
        for my $j (0 .. 2) {
            $coo->[$i][$j] = $r[$j] * $scale;
        }
    }
}

sub kardir {
    my ($coo, $basis, $scale, $n) = @_;
    for my $i (0 .. $n - 1) {
        my @r = @{$coo->[$i]};
        for my $j (0 .. 2) {
            $r[$j] /= $scale;
        }
        my @a;
        for my $j (0 .. 2) {
            my $sum = 0;
            for my $k (0 .. 2) {
                $sum += $r[$k] * $basis->[$j][$k];
            }
            push @a, $sum;
        }
        $coo->[$i] = \@a;
    }
}

1;
